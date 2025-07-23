import sqlite3
import json
from pprint import pprint # For pretty printing dictionary/JSON output

DATABASE_FILE = 'chat_history.db'

def read_messages_from_db():
    """Reads and prints all messages from the chat_history.db."""
    conn = None
    try:
        # Connect to the SQLite database
        # Use check_same_thread=False if you might run this while the FastAPI app is also running
        # Though for a separate read script, it's often not strictly necessary, good for consistency.
        conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
        conn.row_factory = sqlite3.Row # Allows accessing columns by name (e.g., row['content'])
        cursor = conn.cursor()

        # Select all columns from the messages table, ordered by timestamp
        cursor.execute("SELECT * FROM messages ORDER BY timestamp ASC")

        rows = cursor.fetchall()

        if not rows:
            print(f"No messages found in {DATABASE_FILE}.")
            return

        print(f"--- Messages from {DATABASE_FILE} ---")
        for i, row in enumerate(rows):
            print(f"\n--- Message {i+1} ---")
            # Convert row to a dictionary for easier printing
            message_data = dict(row)

            # Try to pretty print JSON fields if they exist
            if 'lc_kwargs' in message_data and message_data['lc_kwargs']:
                try:
                    message_data['lc_kwargs'] = json.loads(message_data['lc_kwargs'])
                except json.JSONDecodeError:
                    pass # Keep as string if not valid JSON

            if 'tool_calls' in message_data and message_data['tool_calls']:
                try:
                    message_data['tool_calls'] = json.loads(message_data['tool_calls'])
                except json.JSONDecodeError:
                    pass # Keep as string if not valid JSON
            
            pprint(message_data)

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    read_messages_from_db()