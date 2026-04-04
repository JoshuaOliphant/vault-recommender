# ABOUTME: HTTP server for the vault recommender — loads model once, serves fast queries.
# ABOUTME: Uses Starlette (ASGI) with uvicorn; started via the CLI 'serve' subcommand.

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from vault_recommender.recommender import VaultRecommender


def _health(request: Request) -> JSONResponse:
    recommender: VaultRecommender = request.app.state.recommender
    return JSONResponse(
        {"status": "ok", "notes_indexed": len(recommender.index.entries)}
    )


def _recommend(request: Request) -> JSONResponse:
    topic = request.query_params.get("topic")
    note = request.query_params.get("note")

    if topic and note:
        return JSONResponse(
            {"error": "Provide topic or note, not both."}, status_code=400
        )
    if not topic and not note:
        return JSONResponse(
            {"error": "Provide a topic or note query parameter."}, status_code=400
        )

    top_k = int(request.query_params.get("top_k", "3"))
    exclude_linked = request.query_params.get("exclude_linked", "").lower() == "true"

    recommender: VaultRecommender = request.app.state.recommender

    if topic:
        results = recommender.similar_to_topic(topic, top_k=top_k)
    else:
        results = recommender.similar_to_note(
            note, top_k=top_k, exclude_linked=exclude_linked
        )

    return JSONResponse([r.to_dict() for r in results])


def _reload(request: Request) -> JSONResponse:
    # When vault_path and index_dir are stored, reload from disk.
    # Otherwise (test injection), just confirm the current state.
    vault_path = getattr(request.app.state, "vault_path", None)
    index_dir = getattr(request.app.state, "index_dir", None)

    if vault_path and index_dir:
        from vault_recommender.recommender import create_recommender

        request.app.state.recommender = create_recommender(vault_path, index_dir)

    recommender: VaultRecommender = request.app.state.recommender
    count = len(recommender.index.entries)
    return JSONResponse({"message": f"Index reloaded. {count} notes indexed."})


def create_app(
    recommender: VaultRecommender | None = None,
    vault_path: Path | None = None,
    index_dir: Path | None = None,
) -> Starlette:
    """Build the ASGI app. Pass a recommender for test injection."""
    routes = [
        Route("/health", _health, methods=["GET"]),
        Route("/recommend", _recommend, methods=["GET"]),
        Route("/reload", _reload, methods=["POST"]),
    ]

    app = Starlette(routes=routes)

    if recommender is not None:
        app.state.recommender = recommender
    elif vault_path and index_dir:
        from vault_recommender.recommender import create_recommender

        app.state.recommender = create_recommender(vault_path, index_dir)
    else:
        raise ValueError("Provide either a recommender or vault_path + index_dir.")

    app.state.vault_path = vault_path
    app.state.index_dir = index_dir

    return app


def run_server(
    recommender: VaultRecommender | None = None,
    vault_path: Path | None = None,
    index_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 7532,
) -> None:
    """Start the HTTP server with uvicorn."""
    import uvicorn

    app = create_app(
        recommender=recommender, vault_path=vault_path, index_dir=index_dir
    )
    uvicorn.run(app, host=host, port=port)
