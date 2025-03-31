import os
import pickle
from typing import Any

from src.exception.duplicate_user_exception import DuplicateUserError


class StorageManager:
    """
    A class to manage storing and retrieving pickle files for a user.

    This class provides CRUD (Create, Read, Update, Delete) operations for pickle files,
    with each file identified by a user_id (sanitized to be filename-friendly).
    """

    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(), "data")

    def get_user_dir(self, user_id: str) -> str:
        """
        Create and return the directory path for a specific user.

        :param user_id: Unique ID of the user
        :return: Path to the user's directory
        """
        user_dir = os.path.join(self.data_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

    async def create(self, user_id: str, filename: str, data: Any) -> str:
        """
        Create a new pickle file for a user.

        :param user_id: Unique ID of the user
        :param filename: Name of the file (without .pkl extension)
        :param data: Data to be stored
        :return: Full path of the created file
        :raises FileExistsError: If file already exists
        """
        # Sanitize filename and get user directory
        user_dir = self.get_user_dir(user_id)

        # Construct full file path
        file_path = os.path.join(user_dir, f"{filename}.pkl")

        # Check if file already exists to prevent overwriting
        if os.path.exists(file_path):
            raise DuplicateUserError(f"File {file_path} already exists")

        # Write data to pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        return file_path

    async def read(self, user_id: str, filename: str) -> Any:
        """
        Read data from a user's pickle file.

        :param user_id: Unique ID of the user
        :param filename: Name of the file (without .pkl extension)
        :return: Data stored in the pickle file
        :raises FileNotFoundError: If file does not exist
        """
        # Sanitize filename and get user directory
        user_dir = self.get_user_dir(user_id)

        # Construct full file path
        file_path = os.path.join(user_dir, f"{filename}.pkl")

        # Read and return data from pickle file
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    async def update(self, user_id: str, filename: str, data: Any) -> str:
        """
        Update an existing pickle file for a user.

        :param user_id: Unique ID of the user
        :param filename: Name of the file (without .pkl extension)
        :param data: New data to be stored
        :return: Full path of the updated file
        :raises FileNotFoundError: If file does not exist
        """
        # Sanitize filename and get user directory
        user_dir = self.get_user_dir(user_id)

        # Construct full file path
        file_path = os.path.join(user_dir, f"{filename}.pkl")

        # Check if file exists before updating
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")

        # Write updated data to pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        return file_path

    async def delete(self, user_id: str, filename: str):
        """
        Delete a pickle file for a user.

        :param user_id: Unique ID of the user
        :param filename: Name of the file (without .pkl extension)
        :return: True if file was deleted, False if file did not exist
        """
        # Sanitize filename and get user directory
        user_dir = self.get_user_dir(user_id)

        # Construct full file path
        file_path = os.path.join(user_dir, f"{filename}.pkl")

        # Delete file if it exists, else raise FileNotFoundError
        os.remove(file_path)

    async def check_data_exists(self, user_id: str, filename: str):
        """
        Check if a file exists under the user's directory

        :param user_id: Unique ID of the user
        :param filename: Name of the file (without an extension)

        :return: True if file exists, False if it doesn't exist
        """
        # Define the full file path
        file_path = os.path.join(self.data_dir, filename)

        return os.path.exists(file_path)

    async def list_files(self, user_id: str) -> list:
        """
        List all pickle files for a specific user.

        :param user_id: Unique ID of the user
        :return: List of pickle filenames
        """
        # Get user directory
        user_dir = self.get_user_dir(user_id)

        # List all .pkl files in the user's directory
        return [f for f in os.listdir(user_dir) if f.endswith('.pkl')]


storage_manager = StorageManager()
