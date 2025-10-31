# agent.py – Tekisho Research Assistant (Supabase + LiveKit Cloud + RAG)
import os
import json
import logging
import requests
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, WorkerOptions, RoomOutputOptions
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation, silero, tavus
from prompts import SESSION_INSTRUCTION, AGENT_INSTRUCTION

# -------------------------------------
# Environment Setup
# -------------------------------------
load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

TAVUS_API_KEY = os.getenv("TAVUS_API_KEY")
REPLICA_ID = os.getenv("REPLICA_ID")
PERSONA_ID = os.getenv("PERSONA_ID")

# Service API URL (combined_server.py runs on port 5001)
SERVICE_API_URL = "http://localhost:5001"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TekishoAgent")


# =====================================
# Agent Definition
# =====================================
class Assistant(Agent):
    """Tekisho AI Agent with RAG-powered responses and DB retrieval."""

    def __init__(self, instructions=AGENT_INSTRUCTION):
        super().__init__(instructions=instructions)
        self.conversation_context = {
            "client_name": None,
            "company": None,
            "industry": None,
            "mail_id": None,
            "phone_no": None,
            "company_summary": None,
            "research_about_company": None,
            "challenges_discussed": [],
            "greeting_done": False,
            "identity_confirmed": False
        }

    # ------------------
    # Function: Search Client in Database
    # ------------------
    @function_tool()
    async def search_client_in_database(self, name: str, company: str) -> str:
        """
        Search for client information in Supabase database based on name and company.
        Returns personalized greeting with company research if found.
        """
        try:
            logger.info(f"🔍 Calling service API: search_client for {name} from {company}")
            
            # Call service.py API
            response = requests.post(
                f"{SERVICE_API_URL}/api/search_client",
                json={"name": name, "company": company},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success"):
                    # Update conversation context if client found
                    if data.get("client_found") and data.get("client_data"):
                        client_data = data["client_data"]
                        self.conversation_context.update({
                            "record_id": client_data.get("record_id"),
                            "client_name": client_data.get("name"),
                            "company": client_data.get("company"),
                            "mail_id": client_data.get("email"),
                            "phone_no": client_data.get("phone"),
                            "company_summary": client_data.get("description"),
                            "research_about_company": client_data.get("industry"),
                            "identity_confirmed": True
                        })
                        logger.info(f"✅ Client found and context updated")
                    
                    return data.get("response", f"Nice to meet you, {name}!")
                else:
                    return f"Nice to meet you, {name}!"
            else:
                logger.error(f"Service API error: {response.status_code}")
                return f"Great to meet you, {name} from {company}!"
                
        except Exception as e:
            logger.error(f"Error calling service API: {e}")
            return f"Nice to meet you, {name}! I'd love to learn more about {company}."

    # ------------------
    # Function: Get Tekisho Solutions (RAG-powered)
    # ------------------
    @function_tool()
    async def get_tekisho_solutions(
        self, 
        challenge: str, 
        industry: str = None,
        wants_specific_metrics: bool = False
    ) -> str:
        """
        Get AI solutions for a specific business challenge using RAG.
        Returns conversational response with metrics formatted for speech.
        """
        try:
            # Track challenges discussed
            self.conversation_context["challenges_discussed"].append(challenge)
            
            # Use industry from context if not provided
            if not industry and self.conversation_context.get("research_about_company"):
                industry = self.conversation_context["research_about_company"]
            
            logger.info(f"🔍 Calling service API: get_solutions for {challenge}")
            
            # Call service.py API
            response = requests.post(
                f"{SERVICE_API_URL}/api/get_solutions",
                json={
                    "challenge": challenge,
                    "industry": industry,
                    "wants_specific_metrics": wants_specific_metrics
                },
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"✅ Got solution from service API")
                    return data.get("solution", "")
            
            # Fallback if API fails
            logger.error(f"Service API error: {response.status_code}")
            return ("Our AI solutions can help you with that challenge. "
                   "Would you like me to connect you with a solution architect?")
            
        except Exception as e:
            logger.error(f"Error calling service API: {e}")
            return ("Our AI solutions typically deliver significant ROI. "
                   "Would you like to discuss specific numbers for your use case?")

    # ------------------
    # Function: Ask Clarifying Question
    # ------------------
    @function_tool()
    async def ask_for_clarification(self, question: str) -> str:
        """
        Ask a clarifying question to better understand the client's needs.
        """
        try:
            logger.info(f"🔍 Calling service API: ask_clarification")
            
            # Call service.py API
            response = requests.post(
                f"{SERVICE_API_URL}/api/ask_clarification",
                json={"question": question},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"✅ Got clarification response from service API")
                    return data.get("response", question)
            
            return question
            
        except Exception as e:
            logger.error(f"Error calling service API: {e}")
            return question

    # ------------------
    # Function: Schedule Follow-up
    # ------------------
    @function_tool()
    async def schedule_followup(self, reason: str = "discuss solutions in detail") -> str:
        """
        Offer to connect the client with a Tekisho expert.
        """
        try:
            name = self.conversation_context.get("client_name", "there")
            company = self.conversation_context.get("company", "your company")
            
            logger.info(f"🔍 Calling service API: schedule_followup")
            
            # Call service.py API
            response = requests.post(
                f"{SERVICE_API_URL}/api/schedule_followup",
                json={
                    "reason": reason,
                    "client_name": name,
                    "company": company
                },
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"✅ Got followup response from service API")
                    return data.get("response", "")
            
            # Fallback
            return (f"I'd love to connect you with one of our solution architects who can {reason} "
                   f"specifically for {company}. Would that be helpful, {name}?")
            
        except Exception as e:
            logger.error(f"Error calling service API: {e}")
            return "Would you like to schedule a follow-up with our team?"

    # ------------------
    # Function: Summarize Conversation
    # ------------------
    @function_tool()
    async def summarize_conversation(self) -> str:
        """
        Provide a summary of what was discussed and next steps.
        """
        try:
            name = self.conversation_context.get("client_name", "you")
            company = self.conversation_context.get("company", "your organization")
            challenges = self.conversation_context.get("challenges_discussed", [])
            
            logger.info(f"🔍 Calling service API: summarize_conversation")
            
            # Call service.py API
            response = requests.post(
                f"{SERVICE_API_URL}/api/summarize_conversation",
                json={
                    "client_name": name,
                    "company": company,
                    "challenges_discussed": challenges
                },
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"✅ Got summary from service API")
                    return data.get("summary", "")
            
            # Fallback
            if challenges:
                summary = (f"It's been great talking with you, {name}! We've discussed how Tekisho can help {company} "
                          f"with {', '.join(challenges[:2])}. ")
            else:
                summary = f"Thank you for sharing about {company}'s goals. "
            
            summary += ("I can connect you with our team to dive deeper into solutions. "
                       "Would you like me to arrange that?")
            return summary
            
        except Exception as e:
            logger.error(f"Error calling service API: {e}")
            return "Thank you for the conversation! Would you like to connect with our team?"

    # ------------------
    # Function: Store Chat History
    # ------------------
    async def store_chat_history(self, chat_messages: list) -> dict:
        """
        Store the complete chat history to Supabase when conversation ends.
        
        Args:
            chat_messages: List of chat messages with timestamp, speaker, and message
            
        Returns:
            Dict with success status and record info
        """
        try:
            # Get client info from conversation context
            client_name = self.conversation_context.get("client_name", "Unknown")
            company_name = self.conversation_context.get("company", "Unknown Company")
            
            logger.info(f"🔍 Calling service API: store_chat_history")
            
            # Call service.py API
            response = requests.post(
                f"{SERVICE_API_URL}/api/store_chat_history",
                json={
                    "client_name": client_name,
                    "company": company_name,
                    "chat_messages": chat_messages
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"✅ Chat history stored successfully via service API")
                    return {"success": True, "record": data}
            
            logger.error(f"Service API error: {response.status_code}")
            return {"success": False, "error": "Failed to store chat history"}
            
        except Exception as e:
            logger.error(f"Error calling service API: {e}")
            return {"success": False, "error": str(e)}


# =====================================
# Entrypoint
# =====================================
async def entrypoint(ctx: agents.JobContext):
    """Main entry for LiveKit agent session."""
    logger.info("Starting Tekisho RAG-Powered Agent with DB Integration...")

    # Create agent with instructions from prompts.py
    agent = Assistant(instructions=AGENT_INSTRUCTION)

    # Set up voice, avatar, and session with transcription support
    session = AgentSession(
        llm="openai/gpt-4o-mini",
        stt="assemblyai/universal-streaming",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
    )

    avatar = tavus.AvatarSession(
        replica_id=REPLICA_ID,
        persona_id=PERSONA_ID,
        api_key=TAVUS_API_KEY,
    )

    try:
        # Start avatar first (Tavus handles audio separately)
        await avatar.start(session, room=ctx.room)
        logger.info("✅ Tavus avatar started")

        # Start session with transcription enabled
        # Tavus handles audio, so we disable audio but enable transcription
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
            room_output_options=RoomOutputOptions(
                audio_enabled=True,  # Tavus handles audio
                transcription_enabled=True,  # Enable LiveKit transcription
            ),
        )
        logger.info("✅ Session started with transcription enabled (audio_enabled=True for Tavus)")

        # Generate initial greeting
        await session.generate_reply(instructions=SESSION_INSTRUCTION)
        
    except Exception as e:
        logger.error(f"Error in agent session: {e}")
        # Ensure avatar session is properly cleaned up on error
        try:
            if hasattr(avatar, 'conversation_id') and avatar.conversation_id:
                logger.info("Cleaning up Tavus avatar session due to error...")
                await avatar.stop()
        except Exception as cleanup_error:
            logger.warning(f"Error during avatar cleanup: {cleanup_error}")
        raise
    finally:
        # Always try to cleanup avatar session when done
        try:
            if hasattr(avatar, 'conversation_id') and avatar.conversation_id:
                logger.info("Cleaning up Tavus avatar session...")
                await avatar.stop()
        except Exception as cleanup_error:
            logger.warning(f"Error during final avatar cleanup: {cleanup_error}")


# =====================================
# CLI Run
# =====================================
if __name__ == "__main__":
    logger.info("Launching Tekisho RAG-Powered Assistant with DB Integration...")
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        ),
    )