import json
import sqlite3

db_path = r"C:\Users\DELL\.local\share\mimocode\mimocode.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

session_id = "ses_00fd00e0bffej569xlE7y4mo9B"

# Get the full pytest output
print("=== ALL TOOL OUTPUTS ===")
c.execute(
    """
    SELECT json_extract(p.data, '$.tool') as tool,
           json_extract(p.data, '$.state.input.command') as cmd,
           json_extract(p.data, '$.state.output') as output
    FROM message m
    JOIN part p ON p.message_id = m.id
    WHERE m.session_id = ?
      AND json_extract(m.data, '$.role') = 'assistant'
      AND json_extract(p.data, '$.type') = 'tool'
    ORDER BY m.time_created
""",
    (session_id,),
)
parts = c.fetchall()
for tool, cmd, output in parts:
    print(f"\nTOOL: {tool}")
    if cmd:
        print(f"CMD: {cmd[:300]}")
    if output:
        out_str = str(output)
        print(f"OUTPUT ({len(out_str)} chars):")
        print(out_str[:4000])
    print("---")

# Check if there were more assistant text messages after tools
print("\n=== ALL ASSISTANT TEXT AFTER TOOLS ===")
c.execute(
    """
    SELECT p.data
    FROM message m
    JOIN part p ON p.message_id = m.id
    WHERE m.session_id = ?
      AND json_extract(m.data, '$.role') = 'assistant'
      AND json_extract(p.data, '$.type') = 'text'
    ORDER BY m.time_created
""",
    (session_id,),
)
for row in c.fetchall():
    data = json.loads(row[0])
    text = data.get("text", "")
    if text:
        print(f"TEXT: {text[:2000]}")
        print()

conn.close()
