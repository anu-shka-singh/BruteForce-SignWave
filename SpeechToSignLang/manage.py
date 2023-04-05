import os
import sys

def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'A2SL.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Make sure you have all the requirements"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
