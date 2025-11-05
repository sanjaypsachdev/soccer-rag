"""Main entry point for the RAG pipeline."""
import argparse

from app.chatbot import run_chatbot
from app.ingestion import run_ingestion


def main():
    """Main function to parse arguments and execute appropriate workflow."""
    parser = argparse.ArgumentParser(
        description="RAG pipeline for soccer data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ingest          Run the ingestion workflow
  python main.py -i                 Run the ingestion workflow (short form)
  python main.py --chat             Start the interactive chatbot
  python main.py -c                 Start the interactive chatbot (short form)
        """
    )
    
    parser.add_argument(
        "--ingest",
        "-i",
        action="store_true",
        help="Run the ingestion workflow to sync vector store with datasets directory"
    )
    
    parser.add_argument(
        "--chat",
        "-c",
        action="store_true",
        help="Run the interactive chatbot to query the vector store"
    )
    
    args = parser.parse_args()
    
    if args.ingest:
        run_ingestion()
    elif args.chat:
        run_chatbot()
    else:
        parser.print_help()
        print("\n⚠️  No action specified. Use --ingest to run ingestion or --chat to start the chatbot.")


if __name__ == "__main__":
    main()
