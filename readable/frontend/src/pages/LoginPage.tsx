import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import toast from "react-hot-toast";

import { login } from "../api/auth";
import { ErrorBanner } from "../components/ErrorBanner";
import { getErrorMessage } from "../lib/errors";
import { authStore } from "../stores/authStore";

export const LoginPage = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("teacher@readable.app");
  const [password, setPassword] = useState("password123");
  const storeLogin = authStore((state) => state.login);

  const mutation = useMutation({
    mutationFn: login,
    onSuccess: (response) => {
      storeLogin(response.access_token, response.user);
      toast.success("Welcome back.");
      navigate("/dashboard");
    },
  });
  const mutationMessage = mutation.isError ? getErrorMessage(mutation.error) : "";
  const looksLikeVoiceFeatureValidation =
    mutationMessage.includes("speech_rate_wps") ||
    mutationMessage.includes("pause_duration_ms") ||
    mutationMessage.includes("pause_frequency") ||
    mutationMessage.includes("mispronunciation_rate") ||
    mutationMessage.includes("repetition_rate");

  return (
    <div className="flex min-h-screen items-center justify-center bg-hero-radial px-4 py-12">
      <div className="grid w-full max-w-5xl gap-8 lg:grid-cols-[1.1fr,0.9fr]">
        <div className="rounded-[2rem] bg-ink p-10 text-white shadow-soft">
          <p className="text-sm uppercase tracking-[0.3em] text-teal-200">Readable</p>
          <h1 className="mt-6 text-4xl font-semibold leading-tight">
            Reading intervention tools shaped for confidence, pacing, and progress.
          </h1>
          <p className="mt-6 max-w-lg text-base text-slate-300">
            This scaffold pairs guided reading workflows with clearly marked AI stubs so we can
            grow into real speech, OCR, and personalization models later.
          </p>
        </div>

        <form
          className="rounded-[2rem] bg-white p-8 shadow-soft"
          onSubmit={(event) => {
            event.preventDefault();
            mutation.mutate({ email, password });
          }}
        >
          <h2 className="text-2xl font-semibold text-ink">Log in</h2>
          <p className="mt-2 text-sm text-slate-500">
            Use the seeded teacher or student credentials after running the backend seed script.
          </p>
          <div className="mt-6 space-y-4">
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-slate-700">Email</span>
              <input
                className="w-full rounded-2xl border border-slate-200 px-4 py-3 outline-none transition focus:border-sea"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
            </label>
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-slate-700">Password</span>
              <input
                type="password"
                className="w-full rounded-2xl border border-slate-200 px-4 py-3 outline-none transition focus:border-sea"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </label>
          </div>
          {mutation.isError ? (
            <div className="mt-4 space-y-2">
              <ErrorBanner
                message={
                  looksLikeVoiceFeatureValidation
                    ? "Login request reached a non-auth endpoint. Check API base URL/server route configuration."
                    : mutationMessage
                }
              />
            </div>
          ) : null}
          <button
            type="submit"
            disabled={mutation.isPending}
            className="mt-6 w-full rounded-2xl bg-sea px-4 py-3 font-semibold text-white transition hover:bg-teal-700 disabled:opacity-60"
          >
            {mutation.isPending ? "Logging in..." : "Log in"}
          </button>
          <p className="mt-4 text-sm text-slate-500">
            Need an account?{" "}
            <Link className="font-medium text-sea" to="/register">
              Register here
            </Link>
          </p>
        </form>
      </div>
    </div>
  );
};
