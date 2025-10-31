# combined_server.py - Unified Flask Server (Token Generation + Service APIs)
import os
import json
import logging
import asyncio
import re
import uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from livekit import api
from livekit.api import LiveKitAPI, ListRoomsRequest
from supabase_client import get_supabase_client, format_chat_message
import rag

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TekishoCombinedServer")


# =====================================
# Helper Functions
# =====================================
def format_numbers_for_speech(text: str) -> str:
    """
    Convert numeric patterns to speech-friendly format.
    Examples:
    - "150-200%" -> "one fifty to two hundred percent"
    - "25-35%" -> "twenty five to thirty five percent"
    - "6-8 weeks" -> "six to eight weeks"
    """
    
    def number_to_words(num_str):
        """Convert number string to words (simple version for common cases)"""
        num = int(num_str)
        
        # Handle special cases
        if num == 0:
            return "zero"
        
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
                 "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        if num < 10:
            return ones[num]
        elif num < 20:
            return teens[num - 10]
        elif num < 100:
            return tens[num // 10] + (" " + ones[num % 10] if num % 10 != 0 else "")
        elif num < 1000:
            hundreds = ones[num // 100] + " hundred"
            remainder = num % 100
            if remainder == 0:
                return hundreds
            elif remainder < 10:
                return hundreds + " and " + ones[remainder]
            elif remainder < 20:
                return hundreds + " and " + teens[remainder - 10]
            else:
                return hundreds + " and " + tens[remainder // 10] + (" " + ones[remainder % 10] if remainder % 10 != 0 else "")
        else:
            return num_str  # Fallback for large numbers
    
    # Pattern for percentage ranges like "150-200%"
    def replace_percent_range(match):
        start = match.group(1)
        end = match.group(2)
        start_words = number_to_words(start)
        end_words = number_to_words(end)
        return f"{start_words} to {end_words} percent"
    
    # Pattern for simple ranges like "6-8 weeks"
    def replace_number_range(match):
        start = match.group(1)
        end = match.group(2)
        unit = match.group(3)
        start_words = number_to_words(start)
        end_words = number_to_words(end)
        return f"{start_words} to {end_words} {unit}"
    
    # Pattern for single percentages like "25%"
    def replace_single_percent(match):
        num = match.group(1)
        num_words = number_to_words(num)
        return f"{num_words} percent"
    
    # Apply replacements
    text = re.sub(r'(\d+)-(\d+)%', replace_percent_range, text)
    text = re.sub(r'(\d+)-(\d+)\s+(weeks|days|months|hours)', replace_number_range, text)
    text = re.sub(r'(\d+)%', replace_single_percent, text)
    
    return text


# =====================================
# LiveKit Token & Room Management
# =====================================
async def generate_room_name():
    """Generate unique room name."""
    name = "room-" + str(uuid.uuid4())[:8]
    rooms = await get_rooms()
    while name in rooms:
        name = "room-" + str(uuid.uuid4())[:8]
    return name

async def get_rooms():
    """Get list of active LiveKit rooms."""
    lk_api = LiveKitAPI()
    rooms = await lk_api.room.list_rooms(ListRoomsRequest())
    await lk_api.aclose()
    return [room.name for room in rooms.rooms]

@app.route("/getToken")
async def get_token():
    """Generate LiveKit access token for room."""
    name = request.args.get("name", "my name")
    room = request.args.get("room", None)
    
    if not room:
        room = await generate_room_name()
        
    token = api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")) \
        .with_identity(name)\
        .with_name(name)\
        .with_grants(api.VideoGrants(
            room_join=True,
            room=room
        ))
    
    return token.to_jwt()


# =====================================
# Service API Endpoints
# =====================================

@app.route("/api/search_client", methods=["POST"])
async def search_client_api():
    """Search for client information in Supabase database."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        name = data.get("name", "").strip()
        company = data.get("company", "").strip()
        
        if not name or not company:
            return jsonify({"error": "Both 'name' and 'company' are required"}), 400
        
        logger.info(f"🔍 Searching for client: {name} from {company}")
        
        supabase_client = get_supabase_client()
        client_doc = await supabase_client.search_client_by_company(company)
        if not client_doc:
            client_doc = await supabase_client.search_client_by_name(name)
        
        if client_doc:
            record_id = str(client_doc.get('id', ''))
            record_company_name = client_doc.get('company_name', '')
            record_name = client_doc.get('name', '')
            record_email = client_doc.get('email', '')
            record_phone = client_doc.get('phone', '')
            record_industry = client_doc.get('industry', '')
            record_description = client_doc.get('description', '')
            
            response = f"Hi {record_name}! "
            if record_company_name:
                response += f"I see you're from {record_company_name}. "
            if record_industry:
                response += f"Your company operates in the {record_industry} industry. "
            if record_description:
                response += f"{record_description} "
            response += "It's wonderful to connect with you! What specific challenges or opportunities can I help you explore today?"
            
            logger.info(f"✅ Found client in database: {record_name} from {record_company_name}")
            
            return jsonify({
                "success": True,
                "client_found": True,
                "response": response,
                "client_data": {
                    "record_id": record_id,
                    "name": record_name,
                    "company": record_company_name,
                    "email": record_email,
                    "phone": record_phone,
                    "industry": record_industry,
                    "description": record_description
                }
            })
        else:
            logger.info(f"❌ Client not found: {name} from {company}")
            response = (f"Nice to meet you, {name}! I don't have prior information about {company} in our system yet, "
                       f"but I'd love to learn more about your business and the challenges you're facing. "
                       f"Could you tell me a bit about what {company} does and what brings you here today?")
            
            return jsonify({
                "success": True,
                "client_found": False,
                "response": response,
                "client_data": None
            })
        
    except Exception as e:
        logger.error(f"Error in search_client_api: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/get_solutions", methods=["POST"])
def get_solutions_api():
    """Get AI solutions for a specific business challenge using RAG."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        challenge = data.get("challenge", "").strip()
        
        if not challenge:
            return jsonify({"error": "'challenge' field is required"}), 400
        
        industry = data.get("industry", None)
        wants_specific_metrics = data.get("wants_specific_metrics", False)
        
        logger.info(f"🔍 Getting solutions for challenge: {challenge} | Industry: {industry}")
        
        try:
            answer = rag.get_tekisho_solutions(
                challenge=challenge, 
                industry=industry
            )
            answer = format_numbers_for_speech(answer)
            
            logger.info(f"✅ Generated solution for challenge: {challenge}")
            
            return jsonify({
                "success": True,
                "challenge": challenge,
                "industry": industry,
                "solution": answer
            })
            
        except Exception as rag_error:
            logger.error(f"RAG function failed: {str(rag_error)}")
            fallback = ("Our AI solutions typically deliver ROI ranging from one fifty to three hundred percent "
                       "within the first six to twelve weeks. Cost savings usually fall between twenty five and forty percent, "
                       "with productivity improvements of fifty to eighty percent. "
                       "Would you like me to connect you with a solution architect to discuss specific numbers for your use case?")
            
            return jsonify({
                "success": True,
                "challenge": challenge,
                "industry": industry,
                "solution": fallback,
                "note": "Using fallback response due to RAG error"
            })
        
    except Exception as e:
        logger.error(f"Error in get_solutions_api: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/ask_clarification", methods=["POST"])
def ask_clarification_api():
    """Ask a clarifying question to better understand the client's needs."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "'question' field is required"}), 400
        
        logger.info(f"❓ Asking clarification: {question}")
        
        return jsonify({
            "success": True,
            "question": question,
            "response": question
        })
        
    except Exception as e:
        logger.error(f"Error in ask_clarification_api: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/schedule_followup", methods=["POST"])
def schedule_followup_api():
    """Offer to connect the client with a Tekisho expert."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        reason = data.get("reason", "discuss solutions in detail").strip()
        client_name = data.get("client_name", "there").strip()
        company = data.get("company", "your company").strip()
        
        logger.info(f"📅 Scheduling follow-up for {client_name} from {company} - Reason: {reason}")
        
        response = (f"I'd love to connect you with one of our solution architects who can {reason} "
                   f"specifically for {company}. They'll provide a customized proposal and answer "
                   f"any technical questions you might have. Would that be helpful, {client_name}?")
        
        logger.info(f"✅ Generated follow-up message for {client_name}")
        
        return jsonify({
            "success": True,
            "reason": reason,
            "client_name": client_name,
            "company": company,
            "response": response
        })
        
    except Exception as e:
        logger.error(f"Error in schedule_followup_api: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/summarize_conversation", methods=["POST"])
def summarize_conversation_api():
    """Provide a summary of what was discussed and next steps."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        client_name = data.get("client_name", "you").strip()
        company = data.get("company", "your organization").strip()
        challenges_discussed = data.get("challenges_discussed", [])
        
        logger.info(f"📝 Summarizing conversation for {client_name} from {company}")
        
        if challenges_discussed and len(challenges_discussed) > 0:
            challenge_text = ', '.join(challenges_discussed[:2])
            summary = (f"It's been great talking with you, {client_name}! We've discussed how Tekisho can help {company} "
                      f"with {challenge_text}. ")
        else:
            summary = f"Thank you for sharing about {company}'s goals. "
        
        summary += ("I can connect you with our team to dive deeper into solutions, "
                   "provide specific ROI calculations, and discuss implementation timelines. "
                   "Would you like me to arrange that?")
        
        logger.info(f"✅ Generated conversation summary for {client_name}")
        
        return jsonify({
            "success": True,
            "client_name": client_name,
            "company": company,
            "challenges": challenges_discussed,
            "summary": summary
        })
        
    except Exception as e:
        logger.error(f"Error in summarize_conversation_api: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/store_chat_history", methods=["POST"])
async def store_chat_history_api():
    """Store the complete chat history to Supabase when conversation ends."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        client_name = data.get("client_name", "Unknown").strip()
        company = data.get("company", "Unknown Company").strip()
        chat_messages = data.get("chat_messages", [])
        
        if not chat_messages:
            return jsonify({"error": "No chat messages provided"}), 400
        
        logger.info(f"💾 Storing chat history for {client_name} from {company} - {len(chat_messages)} messages")
        
        supabase_client = get_supabase_client()
        result = await supabase_client.save_chat_history(
            name=client_name,
            company_name=company,
            chat_history=chat_messages
        )
        
        if "error" in result:
            logger.error(f"Failed to store chat history: {result['error']}")
            return jsonify({"error": result["error"]}), 500
        
        logger.info(f"✅ Successfully stored chat history for {client_name}")
        
        return jsonify({
            "success": True,
            "client_name": client_name,
            "company": company,
            "message_count": len(chat_messages),
            "record_id": result.get("id")
        })
        
    except Exception as e:
        logger.error(f"Error in store_chat_history_api: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/format_numbers", methods=["POST"])
def format_numbers_api():
    """Convert numeric patterns in text to speech-friendly format."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "'text' field is required"}), 400
        
        logger.info(f"🔢 Formatting numbers in text: {text[:50]}...")
        
        formatted_text = format_numbers_for_speech(text)
        
        logger.info(f"✅ Formatted text: {formatted_text[:50]}...")
        
        return jsonify({
            "success": True,
            "original_text": text,
            "formatted_text": formatted_text
        })
        
    except Exception as e:
        logger.error(f"Error in format_numbers_api: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# =====================================
# Legacy Endpoints from server.py
# =====================================

@app.route("/save-conversation", methods=["POST"])
def save_conversation():
    """
    Endpoint to save conversation transcript to database.
    Extracts user name and company using LLM.
    """
    try:
        data = request.get_json()
        
        if not data or 'chat_history' not in data:
            return jsonify({"error": "No chat history provided"}), 400
        
        chat_history = data['chat_history']
        
        if not chat_history:
            return jsonify({"error": "Chat history is empty"}), 400
        
        logger.info(f"📝 Processing conversation save - {len(chat_history)} messages")
        
        # Import from separate extractor module (avoids LiveKit plugin issues)
        from llm_extractor import extract_user_info_from_chat
        
        # Extract user info using LLM (synchronous)
        user_info = extract_user_info_from_chat(chat_history)
        
        name = user_info.get('name', 'Unknown')
        company = user_info.get('company', 'Unknown')
        
        logger.info(f"📊 Extracted info - Name: {name}, Company: {company}")
        
        # Save to Supabase synchronously
        supabase_client = get_supabase_client()
        
        # Call the async function synchronously using asyncio
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Loop is closed")
            except RuntimeError:
                # Create new event loop if none exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            if loop.is_running():
                # If loop is already running, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        supabase_client.save_chat_history(
                            name=name,
                            company_name=company,
                            chat_history=chat_history
                        )
                    ).result()
            else:
                # If loop is not running, run it
                result = loop.run_until_complete(
                    supabase_client.save_chat_history(
                        name=name,
                        company_name=company,
                        chat_history=chat_history
                    )
                )
        except Exception as loop_error:
            logger.error(f"Event loop error: {loop_error}")
            # Fallback: create fresh loop
            new_loop = asyncio.new_event_loop()
            result = new_loop.run_until_complete(
                supabase_client.save_chat_history(
                    name=name,
                    company_name=company,
                    chat_history=chat_history
                )
            )
            new_loop.close()
        
        logger.info(f"💾 Saved conversation - Name: {name}, Company: {company}, Messages: {len(chat_history)}")
        
        return jsonify({
            "success": True,
            "name": name,
            "company": company,
            "message_count": len(chat_history),
            "record_id": result.get('id') if result else None
        })
        
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/save_chat", methods=["POST"])
async def save_chat():
    """
    Save chat history to Supabase when conversation ends.
    
    Expected JSON payload:
    {
        "name": "Client Name",
        "company_name": "Company Name", 
        "chat_history": [
            {
                "timestamp": "2024-01-01T12:00:00",
                "speaker": "User/Agent/System",
                "message": "Chat message content",
                "type": "text"
            }
        ]
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        name = data.get("name", "Unknown")
        company_name = data.get("company_name", "Unknown Company")
        chat_history = data.get("chat_history", [])
        
        if not chat_history:
            return jsonify({"error": "No chat history provided"}), 400
        
        logger.info(f"Received request to save chat for {name} from {company_name} with {len(chat_history)} messages")
        
        # Get Supabase client and save chat
        supabase_client = get_supabase_client()
        result = await supabase_client.save_chat_history(name, company_name, chat_history)
        
        if "error" in result:
            logger.error(f"Failed to save chat: {result['error']}")
            return jsonify({"error": result["error"]}), 500
        
        logger.info(f"Successfully saved chat history for {name}")
        return jsonify({
            "success": True,
            "message": f"Chat history saved successfully for {name}",
            "record_id": result.get("id"),
            "message_count": len(chat_history)
        })
        
    except Exception as e:
        logger.error(f"Error in save_chat endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/get_chats", methods=["GET"])
async def get_chats():
    """
    Retrieve chat history from Supabase.
    
    Query parameters:
    - name: Optional filter by client name
    - company_name: Optional filter by company name
    - limit: Maximum number of records (default 50)
    """
    try:
        # Get query parameters
        name = request.args.get("name")
        company_name = request.args.get("company_name")
        limit = int(request.args.get("limit", 50))
        
        logger.info(f"Retrieving chats with filters: name={name}, company={company_name}, limit={limit}")
        
        # Get Supabase client and retrieve chats
        supabase_client = get_supabase_client()
        chats = await supabase_client.get_chat_history(name, company_name, limit)
        
        logger.info(f"Retrieved {len(chats)} chat records")
        return jsonify({
            "success": True,
            "chats": chats,
            "count": len(chats)
        })
        
    except Exception as e:
        logger.error(f"Error in get_chats endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def root():
    """Root endpoint - shows API is running."""
    return jsonify({
        "service": "Tekisho Combined Server",
        "status": "online",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "token": "/getToken",
            "apis": [
                "/api/search_client",
                "/api/get_solutions",
                "/api/ask_clarification",
                "/api/schedule_followup",
                "/api/summarize_conversation",
                "/api/store_chat_history",
                "/api/format_numbers"
            ]
        }
    })


# =====================================
# Health Check Endpoint
# =====================================
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "Tekisho Combined Server (Token + APIs)"})


# =====================================
# Run Server
# =====================================
if __name__ == "__main__":
    logger.info("🚀 Starting Tekisho Combined Server (Port 5001)")
    logger.info("📍 Endpoints available:")
    logger.info("   - /getToken (LiveKit token)")
    logger.info("   - /api/search_client")
    logger.info("   - /api/get_solutions")
    logger.info("   - /api/ask_clarification")
    logger.info("   - /api/schedule_followup")
    logger.info("   - /api/summarize_conversation")
    logger.info("   - /api/store_chat_history")
    logger.info("   - /api/format_numbers")
    logger.info("   - /health")
    app.run(host="0.0.0.0", port=5001, debug=True)
