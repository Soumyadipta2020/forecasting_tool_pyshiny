# MongoDB helper functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo import MongoClient
import pandas as pd

def mongo_read(table, db, url):
    """
    Read data from MongoDB
    
    Args:
        table (str): Collection name
        db (str): Database name
        url (str): MongoDB connection URL
        
    Returns:
        pandas.DataFrame: Data from MongoDB
    """
    # Connect to MongoDB
    client = MongoClient(url)
    database = client[db]
    collection = database[table]
    
    # Retrieve all documents
    data = list(collection.find({}, {'_id': 0}))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Close connection
    client.close()
    
    return df

def mongo_append(df, table, db, url):
    """
    Append data to MongoDB collection
    
    Args:
        df (pandas.DataFrame): Data to append
        table (str): Collection name
        db (str): Database name
        url (str): MongoDB connection URL
    """
    # Connect to MongoDB
    client = MongoClient(url)
    database = client[db]
    collection = database[table]
    
    # Convert DataFrame to dictionary records
    records = df.to_dict('records')
    
    # Insert the data into MongoDB
    collection.insert_many(records)
    
    # Verify the data was inserted
    retrieved_data = list(collection.find({}, {'_id': 0}))
    retrieved_df = pd.DataFrame(retrieved_data)
    
    if df.shape == retrieved_df.shape:
        print("Success")
    else:
        print("Failure")
    
    # Close connection
    client.close()

def mongo_create(df, table, db, url):
    """
    Create a new MongoDB collection with data
    
    Args:
        df (pandas.DataFrame): Data to insert
        table (str): Collection name
        db (str): Database name
        url (str): MongoDB connection URL
    """
    # Connect to MongoDB
    client = MongoClient(url)
    database = client[db]
    
    # Drop collection if exists
    if table in database.list_collection_names():
        database[table].drop()
    
    # Create collection
    collection = database[table]
    
    # Convert DataFrame to dictionary records
    records = df.to_dict('records')
    
    # Insert the data into MongoDB
    collection.insert_many(records)
    
    # Verify the data was inserted
    retrieved_data = list(collection.find({}, {'_id': 0}))
    retrieved_df = pd.DataFrame(retrieved_data)
    
    if df.shape == retrieved_df.shape:
        print("Success")
    else:
        print("Failure")
    
    # Close connection
    client.close()

def mongo_list(db, url):
    """
    List all collections in a MongoDB database
    
    Args:
        db (str): Database name
        url (str): MongoDB connection URL
        
    Returns:
        list: List of collection names
    """
    # Connect to MongoDB
    client = MongoClient(url)
    database = client[db]
    
    # Get collection names
    collection_names = database.list_collection_names()
    
    # Close connection
    client.close()
    
    return collection_names 
