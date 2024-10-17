import csv

class CSVExporter:
    def __init__(self, filename):
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self._write_header()

    def _write_header(self):
        header = ['frame', 'id', 'x', 'y', 'width', 'height']
        for i in range(10):  # 10 keypoints
            header.extend([f'kp{i}_x', f'kp{i}_y', f'kp{i}_conf'])
        self.csv_writer.writerow(header)

    def write_row(self, row_data):
        self.csv_writer.writerow(row_data)

    def close(self):
        self.csv_file.close()