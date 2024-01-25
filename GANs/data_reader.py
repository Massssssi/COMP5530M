import csv
import numpy as np

"""
Example usage -

data_reader_instance = DataReader("orderbook_snapshots.csv", rows_per_orderbook=2)
data_reader_instance.read_csv()
data_reader_instance.get_data().shape
"""


class DataReader:
    def __init__(self, file_path, rows_per_orderbook=2):
        self.file_path = file_path
        self.rows_per_orderbook = rows_per_orderbook
        self.data = []

    def read_csv(self):
        try:
            with open(self.file_path, "r") as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)
                rows = [[row[2], row[3]] for row in rows]
                self.data = np.array(
                    [
                        rows[i : i + self.rows_per_orderbook]
                        for i in range(0, len(rows), self.rows_per_orderbook)
                    ]
                )
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_data(self):
        return self.data
