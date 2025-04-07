import time
import functools
import csv

class ExecutionLogger:
    def __init__(self):
        self.log = {}

    def add_entry(self, function_name, execution_time):
        self.log[function_name] = execution_time

    def print_log(self):
        for func, time_taken in self.log.items():
            print(f"{func}: {time_taken:.6f} seconds")

    def export_to_csv(self, filename="execution_log.csv"):
        """Exports the log to a CSV file."""
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Function Name", "Execution Time (seconds)"])
            for func, time_taken in self.log.items():
                writer.writerow([func, f"{time_taken:.6f}"])
        print(f"Log exported to {filename}")

def log_execution_time(log_obj_attr):
    """
    Decorator that logs function execution time.

    :param log_obj_attr: Attribute name where the log object is stored (e.g., "log_obj").
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            log_obj = getattr(self, log_obj_attr, None)  # Get log object dynamically
            if log_obj is not None:
                start_time = time.time()
                result = func(self, *args, **kwargs)
                end_time = time.time()
                log_obj.add_entry(func.__name__, end_time - start_time)
                return result
            else:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

