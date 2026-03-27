import snowflake.connector
import toml

secrets = toml.load(".streamlit/secrets.toml")
sf = secrets["snowflake"]

conn = snowflake.connector.connect(
    account=sf["account"],
    user=sf["user"],
    password=sf["password"],
    warehouse=sf["warehouse"],
    database=sf["database"],
    schema=sf["schema"]
)

cursor = conn.cursor()
cursor.execute("SELECT CURRENT_VERSION()")
print("✅ Connected! Snowflake version:", cursor.fetchone()[0])
conn.close()
