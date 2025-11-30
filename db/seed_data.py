import sqlite3
from pathlib import Path
import random
import datetime

DB = Path(__file__).resolve().parents[1] / "db" / "surge.db"
DB.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone TEXT,
    event_name TEXT,
    start_ts TEXT,
    end_ts TEXT,
    impact INTEGER
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS traffic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone TEXT,
    ts TEXT,
    level INTEGER
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS historical_surges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone TEXT,
    ts TEXT,
    surge_level INTEGER
)
""")
conn.commit()

zones = ["Koramangala","Whitefield","MG Road","Indiranagar","Electronic City"]
now = datetime.datetime.now()
for z in zones:
    c.execute("INSERT INTO events(zone,event_name,start_ts,end_ts,impact) VALUES (?,?,?,?,?)",
              (z, f"DemoEvent_{z}", now.isoformat(), (now+datetime.timedelta(hours=4)).isoformat(), random.randint(1,5)))
    for i in range(10):
        ts = (now - datetime.timedelta(hours=i)).isoformat()
        c.execute("INSERT INTO traffic(zone,ts,level) VALUES (?,?,?)", (z,ts, random.randint(1,5)))
    for i in range(24):
        ts = (now - datetime.timedelta(hours=i)).isoformat()
        c.execute("INSERT INTO historical_surges(zone,ts,surge_level) VALUES (?,?,?)", (z,ts, random.choice([0,1,2])))
conn.commit()
conn.close()
print("DB seeded at", DB)
