from pymongo import MongoClient
from config.config import MONGO_URI, DATABASE_NAME, COLLECTION_NAME


def get_collection():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    return collection
