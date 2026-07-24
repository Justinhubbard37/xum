/**
 * TranscriptTimeline — tick marks beside the transcript scrollbar for navigation.
 *
 * From the Figma "Mux exploration" chat dialog exploration (latest feedback round):
 * each user prompt renders a small tick aligned with its relative position in the
 * scroll history; stream errors render wider, destructive-colored ticks. Hovering
 * a tick previews the prompt/error; clicking (or Enter on focus) jumps to it.
 *
 * Positions are measured from the live DOM (`[data-message-id]` anchors) rather
 * than estimated from message counts so ticks stay aligned with real layout. The
 * transcript is not virtualized, so anchors always exist.
 */

import React, { useEffect, useState } from "react";

import { Tooltip, TooltipTrigger, TooltipContent } from "@/browser/components/Tooltip/Tooltip";
import { cn } from "@/common/lib/utils";

export interface TranscriptTimelineMarker {
  historyId: string;
  kind: "prompt" | "error";
  label: string;
}

interface TranscriptTimelineProps {
  /** The transcript scrollport (the element that owns scrollHeight). */
  scrollportRef: React.RefObject<HTMLElement | null>;
  markers: TranscriptTimelineMarker[];
  onNavigate: (historyId: string) => void;
}

interface PositionedMarker {
  marker: TranscriptTimelineMarker;
  /** Marker offset as a fraction (0..1) of the scrollport's scrollHeight. */
  fraction: number;
}

export const TranscriptTimeline: React.FC<TranscriptTimelineProps> = (props) => {
  const { scrollportRef, markers, onNavigate } = props;
  const [positioned, setPositioned] = useState<PositionedMarker[]>([]);

  useEffect(() => {
    const scrollport = scrollportRef.current;
    if (!scrollport) {
      return;
    }

    let frame: number | null = null;

    const measure = () => {
      frame = null;
      const scrollHeight = scrollport.scrollHeight;
      if (scrollHeight <= 0) {
        setPositioned([]);
        return;
      }

      const portTop = scrollport.getBoundingClientRect().top;
      const next: PositionedMarker[] = [];
      for (const marker of markers) {
        const element = scrollport.querySelector(
          `[data-message-id="${CSS.escape(marker.historyId)}"]`
        );
        if (!(element instanceof HTMLElement)) {
          continue;
        }
        const offset = element.getBoundingClientRect().top - portTop + scrollport.scrollTop;
        next.push({ marker, fraction: Math.min(1, Math.max(0, offset / scrollHeight)) });
      }
      setPositioned(next);
    };

    // Coalesce bursts (streaming deltas, resizes) into one layout read per frame.
    const schedule = () => {
      frame ??= requestAnimationFrame(measure);
    };

    schedule();
    const resizeObserver = new ResizeObserver(schedule);
    resizeObserver.observe(scrollport);
    return () => {
      resizeObserver.disconnect();
      if (frame !== null) {
        cancelAnimationFrame(frame);
      }
    };
  }, [scrollportRef, markers]);

  if (positioned.length === 0) {
    return null;
  }

  return (
    <div
      // Sits in the scrollbar gutter, left of the thumb; ticks are the only
      // interactive children. Hidden on narrow/mobile layouts where the gutter
      // is too cramped for pointer targets.
      className="pointer-events-none absolute inset-y-0 right-[12px] z-[5] hidden w-4 md:block"
      data-component="TranscriptTimeline"
    >
      {positioned.map(({ marker, fraction }) => (
        <Tooltip key={marker.historyId}>
          <TooltipTrigger asChild>
            <button
              type="button"
              aria-label={
                marker.kind === "error"
                  ? `Jump to error: ${marker.label}`
                  : `Jump to prompt: ${marker.label}`
              }
              onClick={() => onNavigate(marker.historyId)}
              className="pointer-events-auto absolute right-0 flex h-3 w-full -translate-y-1/2 cursor-pointer items-center justify-end"
              style={{ top: `${fraction * 100}%` }}
            >
              <span
                className={cn(
                  "rounded-full",
                  marker.kind === "error"
                    ? "bg-content-destructive h-[2px] w-2.5"
                    : "bg-content-disabled h-px w-[5px]"
                )}
              />
            </button>
          </TooltipTrigger>
          <TooltipContent side="left" className="max-w-72">
            {marker.kind === "error" ? (
              <span className="text-content-destructive">{marker.label}</span>
            ) : (
              <>
                <span className="font-medium">Prompt:</span> {marker.label}
              </>
            )}
          </TooltipContent>
        </Tooltip>
      ))}
    </div>
  );
};
