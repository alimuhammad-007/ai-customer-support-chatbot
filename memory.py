import sqlite3

conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
cursor = conn.cursor()

# table create
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    message TEXT
)
""")

conn.commit()

def save_message(role, message):
    cursor.execute("INSERT INTO messages (role, message) VALUES (?, ?)", (role, message))
    conn.commit()

def get_messages():
    cursor.execute("SELECT role, message FROM messages")
    return cursor.fetchall()