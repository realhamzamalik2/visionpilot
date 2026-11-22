import type { Metadata } from "next";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans-kr/400.css";
import "@fontsource/ibm-plex-sans-kr/600.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "VisionPilot",
  description: "Ask questions about images and videos. An AI agent picks the vision models, runs them, and shows its work.",
};

// Runs before first paint so the saved or system theme never flashes.
const themeScript = `(() => {
  try {
    const saved = localStorage.getItem("vp-theme");
    const dark = saved ? saved === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.classList.toggle("dark", dark);
  } catch (_) {
    document.documentElement.classList.add("dark");
  }
})();`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="dark h-full" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-full">{children}</body>
    </html>
  );
}
