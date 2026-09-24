# Contributing

Thank you for your interest in Camouflage Breaker.

## Development Setup

1. Clone the repository.
2. Create the Python virtual environment.
3. Install Python dependencies from `requirements.txt`.
4. Configure the required model checkpoints and COD10K dataset locally.
5. Start the FastAPI AI service.
6. Start the Spring Boot backend.
7. Test the application through `http://localhost:8080/`.

## Pull Requests

Before submitting a pull request:

- Keep changes focused and documented.
- Do not commit datasets, model checkpoints, virtual environments, or generated outputs.
- Verify the Python inference service starts successfully.
- Verify the Spring Boot backend starts successfully.
- Test the prediction API when inference-related code changes.
- Update the README when project behavior or setup changes.

## Code Quality

Prefer small, maintainable changes. Preserve existing third-party license and attribution notices.

## Issues

For bugs, include the operating system, relevant logs, reproduction steps, and the affected component when possible.
