"""ASGI entrypoint for the hosted application."""

from web_app import create_app

app = create_app()
