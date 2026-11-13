# Hosting and CI/CD

The Vercel production branch is `main_hosting`. Development stays on the existing
branches; merge reviewed changes into `main_hosting` to release them. The GitHub
default branch does not need to change.

## Runtime

Vercel builds `Dockerfile.vercel` with its Container preset. The image includes
the FastAPI app, both trained ONNX models, both MediaPipe task bundles, the web
interface and reference images. OpenCV's Linux system libraries are installed
explicitly. `app.py` is the ASGI entrypoint; Uvicorn listens on `PORT` (default 80).
The function runs in Singapore (`sin1`) with Fluid compute enabled.

This runs real CPU inference on the server. It does not retrain the models or
establish new accuracy results. The scores displayed by the interface come from
the committed training export metadata, not from a new hosting benchmark.

## Release checks

GitHub Actions builds the production image on pushes and pull requests. It checks
JavaScript syntax, runs the existing feature/notebook tests, verifies readiness
and upload limits, checks motion-state isolation, executes both ONNX models, and
tests HTTP assets, image upload and live WebSocket frames in both recognition
modes. The final image is smoke-tested again to catch missing runtime files.

Vercel's native Git integration handles continuous deployment; no Vercel access
token is stored in GitHub Actions. The production Docker stage depends on a
successful test stage, so the same unit and HTTP/WebSocket checks must pass
during Vercel's own build before it can release an image. Other branches produce
previews. CI does not automatically merge development branches into production.

## Limits and user behavior

- The Hobby function limit is 300 seconds per invocation, including a live
  WebSocket connection. Start Camera again after a connection ends. A new
  connection starts a new sign sequence; the browser transcript remains.
- Uploads are limited to 3 MiB in the browser and the API, leaving room below
  Vercel's 4.5 MB HTTP request limit. Use short, compressed clips.
- Startup can be slower after the function scales to zero. The container runtime
  is a Vercel beta. Hobby compute and transfer quotas still apply.
- Media is processed in memory or temporary files and is not saved as a dataset.
  The transcript remains in the current browser page. Reloading the page clears it.
- Each live viewer has independent J/Z motion history. Model weights are shared
  within an instance; inference is serialized by the registry's existing lock.
- Replace model files through Git and redeploy. Reload Models only reloads files
  already in an instance; it does not upload weights or update other instances.

## Verification and local reproduction

```sh
docker build -f Dockerfile.vercel --target production -t asl-live:hosting .
docker run --rm --entrypoint python asl-live:hosting scripts/smoke_hosting.py --spawn-server
docker run --rm -p 8000:80 asl-live:hosting
```

`GET /healthz` returns 200 only when both models and MediaPipe assets are present.
`GET /api/status` reports the loaded model labels and export metadata. After a
release, verify the public homepage, health, image upload and `/ws/live` against
the production URL. Full signing quality and camera latency need a real webcam
test; synthetic smoke frames only verify the functioning inference pipeline.

Official references: [Vercel containers](https://vercel.com/docs/functions/runtimes/container),
[FastAPI WebSockets](https://vercel.com/docs/functions/websockets),
[function limits](https://vercel.com/docs/functions/limitations).
