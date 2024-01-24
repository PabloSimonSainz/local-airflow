import pandas as pd
from sqlalchemy import create_engine
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime
from etl_local_to_posgres.ingestor.i_ingestor import IIngestor

class IngestorCSVToPostgres(IIngestor):
    def __init__(self, postgres_conn_id, table_name, dataset_path, pk):
        self.postgres_conn_id = postgres_conn_id
        self.table_name = table_name
        self.dataset_path = dataset_path
        self.pk = pk

    def bulk_csv_to_postgres(self) -> list:
        """
        Copy from csv to postgres table.
        
        :return: list of table columns inserted
        :rtype: list
        """
        # Create connection to postgres
        pg_hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)
        engine = create_engine(pg_hook.get_uri())
        
        # Load data from csv to pandas dataframe
        df = pd.read_csv(self.dataset_path, sep=';', decimal=',')
        
        # lowercase column names
        df.columns = df.columns.str.lower()
        
        # Upsert data to postgres
        df.to_sql(f"{self.table_name}_temp", engine, index=False, if_exists='replace')
        
        print(f"Upserted data to {self.table_name}_temp table")
        
        return df.columns.tolist()

    def upsert_data_to_postgres(self, cols: list) -> None:
        """
        Upsert data from postgres table to another postgres table.
        
        :param cols: list of table columns inserted
        :type cols: list
        :return: None
        :rtype: None
        """
        # Create connection to postgres
        pg_hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)
        engine = create_engine(pg_hook.get_uri())
        
        # Upsert data to postgres
        with engine.connect() as conn:
            conn.execute(f"""
                INSERT INTO {self.table_name} ({', '.join(cols)})
                SELECT {', '.join(cols)}
                FROM {self.table_name}_temp
                ON CONFLICT ({self.pk}) DO UPDATE
                SET {', '.join([f"{col}=EXCLUDED.{col}" for col in cols])}
            """)
            
            conn.execute(f"DROP TABLE {self.table_name}_temp")
            
        print(f"Upserted data to {self.table_name} table")