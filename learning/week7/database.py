import sqlite3

connection = None
cursor = None

def init_db():
    global connection, cursor
    connection = sqlite3.connect("learning/week7/detections.db", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    sql_statement = """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        frame INTEGER,
        track_id INTEGER,
        class_name TEXT,
        confidence REAL,
        zone_intrusion BOOLEAN    
    );
    """

    cursor.execute(sql_statement)
    connection.commit()

    return connection

def log_event(timestamp, frame, track_id, class_name, confidence, zone_intrusion):
    cursor.execute(
    "INSERT INTO events (timestamp, frame, track_id, class_name, confidence, zone_intrusion) VALUES (?, ?, ?, ?, ?, ?)",
    (timestamp, frame, track_id, class_name, confidence, zone_intrusion)
    )
    connection.commit()

def get_recent_events(limit):
    cursor.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()  # Returns a list of tuples
    
    return rows
    

