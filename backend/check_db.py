import sqlite3

db_path = r"C:\YuYu\Canon_temp\canon-temp\backend\data\db\factory_test.db"
conn = sqlite3.connect(db_path)

# inspection_log 전체 컬럼 확인
cols = conn.execute("PRAGMA table_info(inspection_log)").fetchall()
print("inspection_log columns:")
for c in cols:
    print(f"  {c[0]}: {c[1]} ({c[2]}) notnull={c[3]} default={c[4]}")

conn.close()
