# VisionPilot web app

Next.js 16 (App Router) + Tailwind CSS v4 + Radix primitives in the shadcn/ui style + Motion (Framer Motion).

- **Viewer**: image or video inside a viewfinder frame; SVG overlays for boxes, masks, pose skeletons, text regions and
  the depth map, drawn in original image pixels so they line up at any size.
- **Live trace**: every agent step streams in over Server-Sent Events as a numbered row with its result and timing.
- **Linked answer**: phrases the agent cites as evidence are hoverable; hovering lights up the exact regions, clicking a
  video moment seeks the player there.
- **Video timeline**: count over time under the player, click to jump to a frame.

```powershell
npm install
npm run dev        # http://localhost:3000 (needs the API on :8000 and the MCP server on :8765)
npm run lint
npm run typecheck
```
