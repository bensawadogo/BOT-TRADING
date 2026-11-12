"""Configuration pytest globale : tests reproductibles, sans le .env local."""
import os
import sys

# Les tests ne lisent jamais le .env local (valeurs par défaut reproductibles).
os.environ["BOT_SKIP_DOTENV"] = "1"
os.environ["TREND_NEWS"] = "0"      # pas d'appel réseau : les tests injectent leur calendrier

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
