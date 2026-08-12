"""Best-effort Gmail notification for the detached behavior benchmark worker."""

from __future__ import annotations

import argparse
import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipient", required=True)
    parser.add_argument("--status", choices=("success", "failure"), required=True)
    parser.add_argument("--scope", choices=("model", "pipeline"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--completed", type=int, default=1)
    parser.add_argument("--total", type=int, default=1)
    args = parser.parse_args()

    sender = os.environ.get("GMAIL_ADDRESS")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender or not app_password:
        print("Skipping email: GMAIL_ADDRESS and/or GMAIL_APP_PASSWORD is missing.")
        return

    try:
        succeeded = args.status == "success"
        status_word = "finished" if succeeded else "failed"
        label = "model" if args.scope == "model" else "pipeline"
        message = EmailMessage()
        message["Subject"] = f"Behavior benchmark {label} {status_word}: {args.name}"
        message["From"] = sender
        message["To"] = args.recipient
        message.set_content(
            f"The behavior benchmark {label} {args.name!r} {status_word}.\n\n"
            f"Progress: {args.completed}/{args.total}\n"
            f"Run directory: {args.run_directory.resolve()}\n"
            + (f"Output directory: {args.output_dir.resolve()}\n" if args.output_dir else "")
        )

        attachments: list[Path] = []
        if args.output_dir and args.output_dir.is_dir():
            attachments.extend([
                args.output_dir / "traditional_metrics.json",
                args.output_dir / "segment_metrics.json",
                args.output_dir / "confusion_matrices.png",
            ])
        attachments.extend([
            args.run_directory / "run_info.txt",
            args.run_directory / "pipeline.log",
        ])
        for path in attachments:
            if not path.is_file():
                continue
            mime, _ = mimetypes.guess_type(path.name)
            maintype, subtype = mime.split("/", 1) if mime else ("application", "octet-stream")
            message.add_attachment(
                path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name
            )

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(message)
        print(f"Sent {label} {args.status} email to {args.recipient}")
    except Exception as error:
        print(f"WARNING: notification email failed ({type(error).__name__}: {error}) -- continuing.")


if __name__ == "__main__":
    main()
