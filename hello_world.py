from dotenv import load_dotenv
import os
load_dotenv()

wrds_user = os.getenv("WRDS_USER")
wrds_password = os.getenv("WRDS_PASSWORD")