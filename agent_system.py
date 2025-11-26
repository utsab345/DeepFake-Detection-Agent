"""
Multi-Agent Deepfake Detection System

This module implements a multi-agent system for detecting deepfakes in images.
It uses:
1. Detection Agent: Runs ML model for deepfake detection
2. Analysis Agent: LLM-powered agent for explaining results
3. Orchestrator: Coordinates the workflow

Key Concepts Implemented (Kaggle Requirements):
- Multi-agent system (Sequential agents)
- Agent powered by LLM (Gemini)
- Custom tools (Deepfake detection tool)
- Sessions & Memory (Conversation history)
- Observability (Logging and tracing)
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import google.generativeai as genai
from colorama import Fore, Style, init

from deepfake_detector import DeepfakeDetector
from config import Config

# Initialize colorama for colored console output
init(autoreset=True)

# Configure logging for observability
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: str  # 'user', 'agent', 'system'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """
    Manages conversation history and session state.
    Implements: Sessions & Memory concept from Kaggle requirements.
    """
    
    def __init__(self, max_history: int = Config.MAX_CONVERSATION_HISTORY):
        self.messages: List[Message] = []
        self.max_history = max_history
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Initialized conversation memory for session: {self.session_id}")
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to conversation history."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        
        # Keep only recent messages to manage context size
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        
        logger.debug(f"Added {role} message to memory")
    
    def get_context(self) -> str:
        """Get formatted conversation context for LLM."""
        context_parts = []
        for msg in self.messages[-5:]:  # Last 5 messages for context
            context_parts.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(context_parts)
    
    def save_session(self, filepath: str = None):
        """Save conversation history to file."""
        if filepath is None:
            filepath = f"session_{self.session_id}.json"
        
        data = {
            "session_id": self.session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in self.messages
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved session to {filepath}")


class DeepfakeDetectionTool:
    """
    Custom tool for deepfake detection.
    Implements: Custom tools concept from Kaggle requirements.
    """
    
    def __init__(self):
        self.detector = DeepfakeDetector()
        logger.info("Initialized DeepfakeDetectionTool")
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Run deepfake detection on an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Detection results with label, confidence, and scores
        """
        logger.info(f"Running detection on: {image_path}")
        
        if Config.ENABLE_TRACING:
            start_time = datetime.now()
        
        result = self.detector.predict(image_path)
        
        if Config.ENABLE_TRACING:
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Detection completed in {duration:.2f}s")
            result['processing_time'] = duration
        
        return result
    
    def get_description(self) -> str:
        """Get tool description for LLM context."""
        return """
        DeepfakeDetectionTool: Analyzes images to detect if they are real or AI-generated deepfakes.
        Returns: label (Realism/Deepfake), confidence score, and detailed probability scores.
        """


class AnalysisAgent:
    """
    LLM-powered agent for analyzing detection results and answering questions.
    Implements: Agent powered by LLM concept from Kaggle requirements.
    """
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL_NAME)
        logger.info(f"Initialized AnalysisAgent with {Config.GEMINI_MODEL_NAME}")
    
    def analyze(self, detection_result: Dict[str, Any], user_query: str = None, 
                conversation_context: str = "") -> str:
        """
        Analyze detection results and provide explanation.
        
        Args:
            detection_result: Results from deepfake detection
            user_query: Optional user question
            conversation_context: Previous conversation for context
            
        Returns:
            Analysis and explanation from the LLM
        """
        logger.info("AnalysisAgent processing request")
        
        # Build prompt for the LLM
        prompt = self._build_prompt(detection_result, user_query, conversation_context)
        
        try:
            response = self.model.generate_content(prompt)
            analysis = response.text
            logger.info("AnalysisAgent completed analysis")
            return analysis
        except Exception as e:
            logger.error(f"AnalysisAgent error: {e}")
            return f"Error during analysis: {str(e)}"
    
    def _build_prompt(self, detection_result: Dict[str, Any], 
                     user_query: str = None, context: str = "") -> str:
        """Build the prompt for the LLM."""
        
        base_prompt = f"""You are an AI assistant specialized in deepfake detection and media forensics.

Detection Results:
- Prediction: {detection_result.get('label', 'Unknown')}
- Confidence: {detection_result.get('confidence', 0):.2%}
- Detailed Scores: {json.dumps(detection_result.get('all_scores', {}), indent=2)}

Previous Conversation Context:
{context}

Your task: """
        
        if user_query:
            base_prompt += f"Answer the user's question: {user_query}"
        else:
            base_prompt += """Provide a clear, concise explanation of the detection results. 
Include:
1. What the result means
2. Confidence level interpretation
3. Potential implications
4. Any caveats or limitations"""
        
        return base_prompt


class DeepfakeDetectionOrchestrator:
    """
    Orchestrator agent that coordinates the multi-agent workflow.
    Implements: Multi-agent system (Sequential agents) from Kaggle requirements.
    """
    
    def __init__(self, gemini_api_key: str):
        # Initialize components
        self.detection_tool = DeepfakeDetectionTool()
        self.analysis_agent = AnalysisAgent(gemini_api_key)
        self.memory = ConversationMemory()
        
        logger.info("Initialized DeepfakeDetectionOrchestrator")
        print(f"{Fore.GREEN}🤖 Deepfake Detection Agent System Initialized{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Session ID: {self.memory.session_id}{Style.RESET_ALL}\n")
    
    def process_image(self, image_path: str, user_query: str = None) -> str:
        """
        Process an image through the multi-agent workflow.
        
        Workflow:
        1. User provides image (and optional question)
        2. Detection Agent runs ML model
        3. Analysis Agent (LLM) explains results
        4. Response returned to user
        5. Conversation stored in memory
        
        Args:
            image_path: Path to image to analyze
            user_query: Optional question from user
            
        Returns:
            Analysis and explanation
        """
        logger.info(f"Processing image: {image_path}")
        
        # Add user message to memory
        user_msg = f"Analyze image: {image_path}"
        if user_query:
            user_msg += f" | Question: {user_query}"
        self.memory.add_message("user", user_msg)
        
        # Step 1: Detection Agent - Run ML model
        print(f"{Fore.YELLOW}🔍 Detection Agent: Analyzing image...{Style.RESET_ALL}")
        detection_result = self.detection_tool.detect(image_path)
        
        if "error" in detection_result:
            error_msg = f"Error: {detection_result['error']}"
            self.memory.add_message("system", error_msg)
            return error_msg
        
        # Log detection results
        self.memory.add_message(
            "system", 
            f"Detection complete: {detection_result['label']} ({detection_result['confidence']:.2%})",
            metadata=detection_result
        )
        
        print(f"{Fore.CYAN}📊 Result: {detection_result['label']} "
              f"(Confidence: {detection_result['confidence']:.2%}){Style.RESET_ALL}")
        
        # Step 2: Analysis Agent - LLM explains results
        print(f"{Fore.YELLOW}🧠 Analysis Agent: Generating explanation...{Style.RESET_ALL}")
        
        conversation_context = self.memory.get_context()
        analysis = self.analysis_agent.analyze(
            detection_result, 
            user_query, 
            conversation_context
        )
        
        # Add analysis to memory
        self.memory.add_message("agent", analysis, metadata=detection_result)
        
        return analysis
    
    def ask_question(self, question: str) -> str:
        """
        Ask a follow-up question about previous detections.
        Uses conversation memory for context.
        """
        logger.info(f"User question: {question}")
        self.memory.add_message("user", question)
        
        print(f"{Fore.YELLOW}🧠 Analysis Agent: Processing question...{Style.RESET_ALL}")
        
        # Get last detection result from memory
        last_detection = None
        for msg in reversed(self.memory.messages):
            if msg.role == "system" and msg.metadata:
                last_detection = msg.metadata
                break
        
        if not last_detection:
            response = "No previous detection results found. Please analyze an image first."
        else:
            context = self.memory.get_context()
            response = self.analysis_agent.analyze(last_detection, question, context)
        
        self.memory.add_message("agent", response)
        return response
    
    def save_session(self):
        """Save the current session."""
        self.memory.save_session()
        print(f"{Fore.GREEN}💾 Session saved{Style.RESET_ALL}")


def main():
    """Main function to run the agent system interactively."""
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"{Fore.RED}❌ Configuration Error: {e}{Style.RESET_ALL}")
        return
    
    # Initialize orchestrator
    orchestrator = DeepfakeDetectionOrchestrator(Config.GEMINI_API_KEY)
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Welcome to the Deepfake Detection Agent System!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    print("Commands:")
    print("  analyze <image_path> - Analyze an image for deepfakes")
    print("  ask <question> - Ask a question about previous results")
    print("  save - Save the current session")
    print("  quit - Exit the system\n")
    
    while True:
        try:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                orchestrator.save_session()
                print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
                break
            
            elif user_input.lower() == 'save':
                orchestrator.save_session()
            
            elif user_input.lower().startswith('analyze '):
                image_path = user_input[8:].strip()
                print()
                response = orchestrator.process_image(image_path)
                print(f"\n{Fore.GREEN}Agent: {Style.RESET_ALL}{response}\n")
            
            elif user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
                print()
                response = orchestrator.ask_question(question)
                print(f"\n{Fore.GREEN}Agent: {Style.RESET_ALL}{response}\n")
            
            else:
                print(f"{Fore.RED}Unknown command. Use: analyze, ask, save, or quit{Style.RESET_ALL}\n")
        
        except KeyboardInterrupt:
            print(f"\n\n{Fore.CYAN}Interrupted. Saving session...{Style.RESET_ALL}")
            orchestrator.save_session()
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
