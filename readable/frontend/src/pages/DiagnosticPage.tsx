import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import toast from "react-hot-toast";

import { useGazeInput, type GazeSource } from "../hooks/useGazeInput";
import { startDiagnostic, submitDiagnostic } from "../api/sessions";
import { ErrorBanner } from "../components/ErrorBanner";
import { ScoreCard } from "../components/ScoreCard";
import { TextReader } from "../components/TextReader";
import { getErrorMessage } from "../lib/errors";
import { profileStore } from "../stores/profileStore";
import { sessionStore } from "../stores/sessionStore";
import type { GazeFlowSample } from "../types/eyeTracking";

const sentenceSplitPattern = /(?<=[.!?])\s+/;

const buildParagraphs = (passage: string): string[][] => {
  const sentences = passage
    .split(sentenceSplitPattern)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

  const grouped: string[][] = [];
  for (let index = 0; index < sentences.length; index += 2) {
    const chunk = sentences.slice(index, index + 2).join(" ");
    grouped.push(chunk.split(/\s+/).filter(Boolean));
  }

  return grouped;
};

const screenToViewport = (sample: GazeFlowSample): { x: number; y: number } => {
  const horizontalChrome = Math.max(window.outerWidth - window.innerWidth, 0);
  const verticalChrome = Math.max(window.outerHeight - window.innerHeight, 0);
  const sideBorder = horizontalChrome / 2;
  const topChrome = Math.max(verticalChrome - sideBorder, 0);

  return {
    x: sample.GazeX - window.screenX - sideBorder,
    y: sample.GazeY - window.screenY - topChrome,
  };
};

export const DiagnosticPage = () => {
  const {
    currentSession,
    sessionResults,
    eyeTrackingFocusEvents,
    setCurrentSession,
    setSessionResults,
    addEyeTrackingFocusEvent,
    clearEyeTrackingFocusEvents,
  } = sessionStore();
  const setStudentProfile = profileStore((state) => state.setStudentProfile);
  const studentProfile = profileStore((state) => state.studentProfile);
  const gazeSource = (import.meta.env.VITE_GAZE_SOURCE ?? "mouse") as GazeSource;
  const [appKey] = useState(
    import.meta.env.VITE_READABLE_EYE_TRACKER_APP_KEY ??
    import.meta.env.VITE_GAZEFLOW_APP_KEY ??
    "AppKeyTrial",
  );
  const [port] = useState(
    import.meta.env.VITE_READABLE_EYE_TRACKER_PORT ??
    import.meta.env.VITE_GAZEFLOW_PORT ??
    "43333",
  );
  const [activeWordIndex, setActiveWordIndex] = useState<number | null>(null);
  const [gazeDot, setGazeDot] = useState<{ x: number; y: number } | null>(null);
  const [focusedWordCounts, setFocusedWordCounts] = useState<Record<number, number>>({});
  const passageRef = useRef<HTMLDivElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const tracker = useGazeInput({
    source: gazeSource,
    appKey,
    port: Number.parseInt(port, 10) || 43333,
  });

  const startMutation = useMutation({
    mutationFn: startDiagnostic,
    onSuccess: (response) => {
      setCurrentSession({
        sessionId: response.session_id,
        sessionType: "diagnostic",
        expectedText: response.expected_text,
      });
      setSessionResults(null);
      setActiveWordIndex(null);
      setGazeDot(null);
      setFocusedWordCounts({});
      clearEyeTrackingFocusEvents();
      tracker.clearSamples();
      toast.success("Diagnostic passage ready.");
    },
  });

  const submitMutation = useMutation({
    mutationFn: ({ file, eyePayload }: { file: File; eyePayload: Record<string, unknown> }) =>
      submitDiagnostic(currentSession?.sessionId ?? -1, file, eyePayload),
    onSuccess: (response) => {
      setSessionResults(response.result);
      setStudentProfile(response.profile);
      // Force a new diagnostic session next time instead of reusing a stale session id.
      setCurrentSession(null);
      clearEyeTrackingFocusEvents();
      setActiveWordIndex(null);
      setGazeDot(null);
      setFocusedWordCounts({});
      tracker.clearSamples();
      toast.success("Diagnostic submitted.");
    },
  });

  const passage = currentSession?.expectedText ?? "";
  const passageParagraphs = useMemo(() => buildParagraphs(passage), [passage]);
  const passageWords = useMemo(
    () => passage.split(/\s+/).filter(Boolean).map((word) => word.replace(/[.,!?]/g, "")),
    [passage],
  );
  const topFocusedWords = useMemo(
    () =>
      Object.entries(focusedWordCounts)
        .map(([index, count]) => ({
          word: passageWords[Number(index)] ?? `Word ${Number(index) + 1}`,
          count,
        }))
        .sort((left, right) => right.count - left.count)
        .slice(0, 5),
    [focusedWordCounts, passageWords],
  );
  const modelScores = studentProfile?.model_profile_scores ?? {};
  const hasModelScores = Object.keys(modelScores).length > 0;

  useEffect(() => {
    if (!tracker.latestSample || !passageRef.current) {
      return;
    }

    const viewport = screenToViewport(tracker.latestSample);
    const rect = passageRef.current.getBoundingClientRect();
    const insidePassage =
      viewport.x >= rect.left &&
      viewport.x <= rect.right &&
      viewport.y >= rect.top &&
      viewport.y <= rect.bottom;

    setGazeDot(
      insidePassage
        ? {
          x: viewport.x - rect.left,
          y: viewport.y - rect.top,
        }
        : null,
    );

    const element = document.elementFromPoint(viewport.x, viewport.y) as HTMLElement | null;
    const wordElement = element?.closest<HTMLElement>("[data-word-index]");

    if (!wordElement) {
      return;
    }

    const wordIndex = Number(wordElement.dataset.wordIndex);
    if (Number.isNaN(wordIndex)) {
      return;
    }

    setActiveWordIndex(wordIndex);
    addEyeTrackingFocusEvent({
      wordIndex,
      timestamp: tracker.latestSample.receivedAt,
    });
    setFocusedWordCounts((current) => ({
      ...current,
      [wordIndex]: (current[wordIndex] ?? 0) + 1,
    }));
  }, [addEyeTrackingFocusEvent, tracker.latestSample]);

  const buildEyeTrackingPayload = (): Record<string, unknown> => {
    const samples = tracker.samples.slice(-180).map((sample) => ({
      ...sample,
      viewport: screenToViewport(sample),
    }));

    return {
      provider: "readable_local_eye_tracker",
      source: "local_webcam_service",
      authorization_status: tracker.authorizationStatus,
      connection_status: tracker.status,
      sample_count: tracker.samples.length,
      focused_word_hits: topFocusedWords,
      focus_events: eyeTrackingFocusEvents,
      active_word_index: activeWordIndex,
      screen_metrics: {
        screen_x: window.screenX,
        screen_y: window.screenY,
        inner_width: window.innerWidth,
        inner_height: window.innerHeight,
        outer_width: window.outerWidth,
        outer_height: window.outerHeight,
      },
      samples,
    };
  };

  const startTest = async () => {
    if (!currentSession?.sessionId) {
      toast.error("Start a diagnostic session first.");
      return;
    }
    if (isRecording || submitMutation.isPending) {
      return;
    }

    tracker.connect();
    if (!navigator.mediaDevices || typeof MediaRecorder === "undefined") {
      setIsRecording(true);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const file = new File([blob], "reading-session.webm", { type: "audio/webm" });
        mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
        setIsRecording(false);
        await submitMutation.mutateAsync({ file, eyePayload: buildEyeTrackingPayload() });
        tracker.disconnect();
      };

      recorder.start();
      setIsRecording(true);
    } catch {
      setIsRecording(true);
      toast("Microphone access failed, using a simulated recording.");
    }
  };

  const stopTest = async () => {
    if (!isRecording || submitMutation.isPending) {
      return;
    }
    if (!mediaRecorderRef.current) {
      const fallbackFile = new File(["mock audio"], "reading-session.webm", { type: "audio/webm" });
      setIsRecording(false);
      await submitMutation.mutateAsync({ file: fallbackFile, eyePayload: buildEyeTrackingPayload() });
      tracker.disconnect();
      return;
    }
    mediaRecorderRef.current.stop();
  };

  return (
    <div className="space-y-6">
      {!passage ? (
        <section className="rounded-[2rem] bg-hero-radial p-8 shadow-soft">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-sm uppercase tracking-[0.25em] text-sea">Diagnostic Session</p>
              <h1 className="mt-2 text-3xl font-semibold text-ink">Baseline reading check-in</h1>
              <p className="mt-3 max-w-2xl text-slate-600">
                Start a session, read the passage aloud, and review mock speech and attention
                feedback.
              </p>
            </div>
            <button
              type="button"
              onClick={() => startMutation.mutate()}
              className="rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
            >
              {startMutation.isPending ? "Preparing..." : "Start Diagnostic"}
            </button>
          </div>
        </section>
      ) : null}

      {startMutation.isError ? <ErrorBanner message={getErrorMessage(startMutation.error)} /> : null}
      {submitMutation.isError ? <ErrorBanner message={getErrorMessage(submitMutation.error)} /> : null}

      {passage ? (
        <section className="fixed inset-x-0 bottom-0 top-[82px] overflow-hidden bg-[linear-gradient(180deg,#fffaf5_0%,#fff7ed_42%,#f8fafc_100%)]">
          <div className="flex h-full flex-col px-4 pb-4 pt-3 sm:px-6 lg:px-8">
            <div className="grid gap-3 rounded-[1.75rem] bg-white/88 p-4 shadow-soft backdrop-blur lg:grid-cols-[1.15fr,0.85fr]">
              <div className="min-w-0">
                <p className="text-sm uppercase tracking-[0.25em] text-sea">Readable Eye Tracker</p>
                <p className="mt-2 text-sm text-slate-600">
                  Full-page reading mode keeps the paragraph large, centered, and stable for local webcam tracking.
                </p>
              </div>
              <div className="flex flex-wrap items-center justify-start gap-3 lg:justify-end">
                <div className="rounded-full bg-mist px-4 py-2 text-sm font-medium text-sea">
                  Status: {tracker.status}
                </div>
                <div className="rounded-full bg-blush px-4 py-2 text-sm font-medium text-ink">
                  Samples: {tracker.samples.length}
                </div>
                <button
                  type="button"
                  onClick={() => void startTest()}
                  className="rounded-full bg-sea px-5 py-3 text-sm font-semibold text-white transition hover:bg-teal-700"
                >
                  {isRecording ? "Test Running..." : "Start Test"}
                </button>
                <button
                  type="button"
                  onClick={() => void stopTest()}
                  className="rounded-full border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-600 transition hover:border-sea hover:text-sea"
                >
                  Stop Test
                </button>
              </div>
            </div>

            {tracker.error ? <div className="mt-3"><ErrorBanner message={tracker.error} /></div> : null}

            <div className="mt-3 grid min-h-0 flex-1 gap-3 lg:grid-cols-[1fr,320px]">
              <div
                ref={passageRef}
                className="relative min-h-0 overflow-y-auto rounded-[2rem] border border-white/60 bg-white/72 px-6 py-6 shadow-soft backdrop-blur sm:px-10 sm:py-8 lg:px-14 lg:py-10"
              >
                {gazeDot ? (
                  <div
                    className="pointer-events-none absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-sea bg-sea/20"
                    style={{ left: `${gazeDot.x}px`, top: `${gazeDot.y}px` }}
                  />
                ) : null}

                <div className="flex min-h-full items-start justify-center py-2">
                  <div className="w-full max-w-6xl space-y-8 text-center text-[clamp(1.9rem,3vw,3rem)] leading-[2.35] tracking-[0.012em] text-ink">
                    {(() => {
                      let globalWordIndex = 0;
                      return passageParagraphs.map((paragraph, paragraphIndex) => (
                        <p
                          key={`${paragraphIndex}-${paragraph[0] ?? "paragraph"}`}
                          className="mx-auto max-w-[36ch] lg:max-w-[40ch]"
                        >
                          {paragraph.map((word) => {
                            const currentIndex = globalWordIndex;
                            globalWordIndex += 1;

                            return (
                              <span
                                key={`${currentIndex}-${word}`}
                                data-word-index={currentIndex}
                                className={`mx-[0.14em] my-[0.06em] inline-flex max-w-full items-center justify-center rounded-xl px-[0.2em] py-[0.1em] align-baseline break-words transition ${activeWordIndex === currentIndex
                                    ? "bg-sea text-white"
                                    : "bg-white/60"
                                  }`}
                              >
                                {word}
                              </span>
                            );
                          })}
                        </p>
                      ));
                    })()}
                  </div>
                </div>
              </div>

              <aside className="min-h-0 overflow-hidden rounded-[2rem] border border-white/60 bg-white/88 p-4 shadow-soft backdrop-blur">
                <div className="grid gap-3">
                  <div className="rounded-2xl bg-mist p-4">
                    <p className="text-sm text-slate-500">Local authorization</p>
                    <p className="mt-2 font-semibold text-ink">
                      {tracker.authorizationStatus ?? "Waiting for first server message"}
                    </p>
                  </div>

                  <div className="rounded-2xl bg-slate-50 p-4 text-sm text-slate-600">
                    <p className="font-semibold text-ink">Latest local gaze packet</p>
                    <p className="mt-2">
                      Gaze: {tracker.latestSample ? `${tracker.latestSample.GazeX}, ${tracker.latestSample.GazeY}` : "--"}
                    </p>
                    <p className="mt-1">
                      Head pose: {tracker.latestSample ? `${tracker.latestSample.HeadX}, ${tracker.latestSample.HeadY}, ${tracker.latestSample.HeadZ}` : "--"}
                    </p>
                  </div>

                  <div className="rounded-2xl bg-white p-4 ring-1 ring-slate-100 text-sm text-slate-600">
                    <p className="font-semibold text-ink">Recommended setup</p>
                    <p className="mt-2">Sit centered in front of the webcam and keep your face fully lit.</p>
                    <p className="mt-2">Readable uses large type, generous line spacing, and a no-scroll reading canvas for tracking stability.</p>
                  </div>

                  {topFocusedWords.length > 0 ? (
                    <div className="rounded-2xl bg-amber-50 p-4 text-sm text-slate-700">
                      <p className="font-semibold text-ink">Most-fixated words</p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {topFocusedWords.map((item) => (
                          <span
                            key={`${item.word}-${item.count}`}
                            className="rounded-full bg-white px-3 py-2 text-sm font-medium text-amber-900"
                          >
                            {item.word} x{item.count}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              </aside>
            </div>
          </div>
        </section>
      ) : null}

      {sessionResults ? (
        <section className="space-y-6">
          <ScoreCard
            accuracy={sessionResults.accuracy_pct}
            wpm={sessionResults.speed_wpm}
            attention={sessionResults.attention_score}
          />
          <div>
            <h2 className="mb-3 text-xl font-semibold text-ink">Highlighted passage</h2>
            <TextReader text={[sessionResults.expected_text]} highlights={sessionResults.errors} />
          </div>
          <div className="rounded-3xl border border-white/70 bg-white/90 p-6 shadow-soft">
            <h2 className="text-xl font-semibold text-ink">ML Output Profile</h2>
            {hasModelScores ? (
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {Object.entries(modelScores).map(([name, score]) => (
                  <div key={name} className="rounded-2xl bg-mist p-4">
                    <p className="text-sm text-slate-500">{name.replaceAll("_", " ")}</p>
                    <p className="mt-2 text-2xl font-semibold text-ink">{score.toFixed(3)}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-3 text-sm text-slate-600">
                Model scores unavailable. Add `dyslexia_profiler.pt` in `backend/profile_model` and rerun the
                diagnostic test.
              </p>
            )}
          </div>
        </section>
      ) : null}
    </div>
  );
};
