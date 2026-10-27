# Security

Do not submit credentials or private driving data in issues. Report sensitive issues privately through the repository owner's available GitHub contact options.

Runtime configuration and backend commands are trusted local inputs. Tool input JSON is schema-checked and is never executed as shell text. Optional engines may execute Python configuration files; only use trusted configs and checkpoints. The portable screening format uses JSON, not pickle. Model calls transmit explicitly supplied evidence to the configured endpoint. Avoid sharing private runtime logs.
