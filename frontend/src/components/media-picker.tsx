"use client";

import { Film, Loader2, Upload } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { getSamples, type MediaInfo, mediaSrc, type SampleInfo, uploadMedia } from "@/lib/api";
import { cn } from "@/lib/utils";

const SCENARIO_COPY: Record<string, { title: string; blurb: string }> = {
  retail: { title: "Retail", blurb: "Count shoppers, watch the queue over time, spot empty shelves." },
  seoul: { title: "Seoul streets", blurb: "Read Korean signs and menus, translate them, describe the scene." },
  classroom: { title: "Meetings and classrooms", blurb: "Faces, apparent expressions and engagement over time." },
};

const ACCEPT = "image/jpeg,image/png,image/webp,image/bmp,video/mp4,video/quicktime,video/webm";

interface Props {
  onPick: (media: MediaInfo, suggestions: string[]) => void;
}

export function MediaPicker({ onPick }: Props) {
  const [samples, setSamples] = useState<SampleInfo[] | null>(null);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getSamples()
      .then(setSamples)
      .catch((e: Error) => setSampleError(e.message));
  }, []);

  const upload = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      setUploading(true);
      setUploadError(null);
      try {
        onPick(await uploadMedia(file), []);
      } catch (e) {
        setUploadError(e instanceof Error ? e.message : String(e));
      } finally {
        setUploading(false);
      }
    },
    [onPick],
  );

  const grouped = (samples ?? []).reduce<Record<string, SampleInfo[]>>((acc, s) => {
    (acc[s.scenario] ??= []).push(s);
    return acc;
  }, {});

  return (
    <section className="flex flex-col gap-6" aria-label="Choose media">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void upload(e.dataTransfer.files[0]);
        }}
        className={cn(
          "flex flex-col items-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors",
          dragging ? "border-focus bg-focus-soft" : "border-line bg-stage",
        )}
      >
        <div className="flex size-12 items-center justify-center rounded-full bg-panel text-muted">
          {uploading ? <Loader2 className="size-5 animate-spin" /> : <Upload className="size-5" />}
        </div>
        <div>
          <p className="font-semibold">Drop an image or video here</p>
          <p className="text-sm text-muted">JPEG, PNG, WebP, MP4, MOV or WebM, up to 100 MB. Files stay on your machine.</p>
        </div>
        <button
          className="rounded-md bg-focus px-4 py-2 text-sm font-medium text-[#14213a] hover:brightness-105 disabled:opacity-50"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? "Uploading…" : "Choose a file"}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="sr-only"
          onChange={(e) => void upload(e.target.files?.[0])}
        />
        {uploadError && <p className="text-sm text-danger">Upload failed: {uploadError}</p>}
      </div>

      <div className="flex flex-col gap-5">
        <h2 className="text-base font-semibold">Or try a sample</h2>
        {sampleError && (
          <p className="text-sm text-danger">
            Samples could not load ({sampleError}). Start the API with <code>uv run visionpilot-api</code>.
          </p>
        )}
        {samples && samples.length === 0 && (
          <p className="text-sm text-muted">No samples yet. Put images or videos in the samples folder to show them here.</p>
        )}
        {Object.entries(grouped).map(([scenario, items]) => {
          const copy = SCENARIO_COPY[scenario] ?? { title: scenario, blurb: "" };
          return (
            <div key={scenario}>
              <h3 className="text-sm font-semibold">{copy.title}</h3>
              {copy.blurb && <p className="mb-2 text-sm text-muted">{copy.blurb}</p>}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {items.map((sample) => (
                  <button
                    key={sample.id}
                    onClick={() => onPick(sample, sample.suggested_questions)}
                    className="group overflow-hidden rounded-lg border border-line bg-panel text-left transition-colors hover:border-focus"
                  >
                    <div className="relative aspect-video overflow-hidden bg-stage">
                      {sample.kind === "image" ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={mediaSrc(sample)} alt="" className="size-full object-cover" loading="lazy" />
                      ) : (
                        <video src={`${mediaSrc(sample)}#t=0.5`} className="size-full object-cover" muted preload="metadata" />
                      )}
                      {sample.kind === "video" && (
                        <span className="absolute bottom-1.5 left-1.5 flex items-center gap-1 rounded bg-black/60 px-1.5 py-0.5 text-[11px] text-white">
                          <Film className="size-3" /> Video
                        </span>
                      )}
                    </div>
                    <p className="truncate px-2.5 py-2 text-sm">{sample.title}</p>
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
