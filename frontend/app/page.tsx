"use client";

import { useState, useRef, useCallback } from "react";
import ParameterPanel from "@/components/ParameterPanel";
import PossibilityMap from "@/components/PossibilityMap";
import type {
  ThetaParams,
  ResponseResult,
  VariantBundle,
  PossibilityMap as PossibilityMapType,
  ScoredResponse,
  UmapCoord,
} from "@/components/types";

type LoadingAction = "generate" | "variants" | "explore" | null;
type Tab = "response" | "variants" | "map" | "json";

const DEFAULT_THETA: ThetaParams = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 40,
  persona: "balanced",
  domain: "general",
  logit_bias: {},
  repetition_penalty: 1.1,
};

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [theta, setTheta] = useState<ThetaParams>(DEFAULT_THETA);
  const [budget, setBudget] = useState(100);
  const [loading, setLoading] = useState<LoadingAction>(null);
  const [activeTab, setActiveTab] = useState<Tab>("response");

  const [response, setResponse] = useState<ResponseResult | null>(null);
  const [variantBundle, setVariantBundle] = useState<VariantBundle | null>(null);
  const [variantEvents, setVariantEvents] = useState<
    { index: number; label: string; text: string }[]
  >([]);
  const [exploreResult, setExploreResult] = useState<PossibilityMapType | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setLoading(null);
  }, []);

  const doGenerate = useCallback(async () => {
    cancel();
    setLoading("generate");
    setResponse(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, theta }),
        signal: ctrl.signal,
      });
      const data: ResponseResult = await res.json();
      setResponse(data);
      setActiveTab("response");
    } catch (e: any) {
      if (e.name !== "AbortError") console.error(e);
    } finally {
      if (!ctrl.signal.aborted) setLoading(null);
    }
  }, [prompt, theta, cancel]);

  const doVariants = useCallback(async () => {
    cancel();
    setLoading("variants");
    setVariantBundle(null);
    setVariantEvents([]);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const res = await fetch("/api/variants", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
        signal: ctrl.signal,
      });
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith("data: ") && eventType) {
            try {
              const parsed = JSON.parse(line.slice(6));
              if (eventType === "variant") {
                setVariantEvents((prev) => [
                  ...prev,
                  {
                    index: parsed.index,
                    label: parsed.label,
                    text: parsed.text,
                  },
                ]);
                setActiveTab("variants");
              } else if (eventType === "complete") {
                setVariantBundle(parsed as VariantBundle);
              }
            } catch {
              // skip malformed
            }
            eventType = "";
          }
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") console.error(e);
    } finally {
      if (!ctrl.signal.aborted) setLoading(null);
    }
  }, [prompt, cancel]);

  const doExplore = useCallback(async () => {
    cancel();
    setLoading("explore");
    setExploreResult(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const res = await fetch("/api/explore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          budget,
          top_p: theta.top_p,
          max_branch: 4,
          max_depth: 6,
        }),
        signal: ctrl.signal,
      });
      const data: PossibilityMapType = await res.json();
      setExploreResult(data);
      setActiveTab("map");
    } catch (e: any) {
      if (e.name !== "AbortError") console.error(e);
    } finally {
      if (!ctrl.signal.aborted) setLoading(null);
    }
  }, [prompt, budget, theta.top_p, cancel]);

  const handleAction = useCallback(
    (action: string) => {
      if (action === "generate") doGenerate();
      else if (action === "variants") doVariants();
      else if (action === "explore") doExplore();
    },
    [doGenerate, doVariants, doExplore]
  );

  return (
    <div className="flex h-screen overflow-hidden">
      <div className="w-[30%] min-w-[260px] max-w-[340px] p-5 overflow-y-auto border-r border-border">
        <ParameterPanel
          prompt={prompt}
          theta={theta}
          budget={budget}
          loading={loading}
          onChange={(patch) => setTheta((t) => ({ ...t, ...patch }))}
          onPromptChange={setPrompt}
          onBudgetChange={setBudget}
          onAction={handleAction}
        />
      </div>

      <div className="flex-1 flex flex-col overflow-hidden">
        <TabBar activeTab={activeTab} onTabChange={setActiveTab} loading={loading} onCancel={cancel} />
        <div className="flex-1 overflow-y-auto p-5">
          {activeTab === "response" && <ResponseTab response={response} />}
          {activeTab === "variants" && (
            <VariantsTab
              events={variantEvents}
              bundle={variantBundle}
            />
          )}
          {activeTab === "map" && (
            <PossibilityMapTab result={exploreResult} />
          )}
          {activeTab === "json" && <JsonTab response={response} variantBundle={variantBundle} exploreResult={exploreResult} />}
        </div>
      </div>
    </div>
  );
}

function TabBar({
  activeTab,
  onTabChange,
  loading,
  onCancel,
}: {
  activeTab: Tab;
  onTabChange: (t: Tab) => void;
  loading: LoadingAction;
  onCancel: () => void;
}) {
  const tabs: { key: Tab; label: string }[] = [
    { key: "response", label: "Response" },
    { key: "variants", label: "Variants" },
    { key: "map", label: "Map" },
    { key: "json", label: "Raw JSON" },
  ];
  return (
    <div className="flex items-center border-b border-border px-5">
      {tabs.map((t) => (
        <button
          key={t.key}
          onClick={() => onTabChange(t.key)}
          className={`px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === t.key
              ? "text-accent border-accent"
              : "text-gray-500 border-transparent hover:text-gray-300"
          }`}
        >
          {t.label}
        </button>
      ))}
      <div className="flex-1" />
      {loading && (
        <button
          onClick={onCancel}
          className="text-xs text-red-400 hover:text-red-300 px-3 py-1"
        >
          Cancel
        </button>
      )}
    </div>
  );
}

function ResponseTab({ response }: { response: ResponseResult | null }) {
  if (!response) {
    return <EmptyState message="Click Generate Single to see a response" />;
  }
  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-3 text-xs text-gray-400">
        <Badge label="PPL" value={response.perplexity.toFixed(2)} />
        <Badge label="Entropy" value={response.mean_entropy.toFixed(2)} />
        <Badge label="Tokens" value={String(response.token_count)} />
      </div>
      <div className="bg-surface border border-border rounded p-4 whitespace-pre-wrap text-sm leading-relaxed">
        {response.text || <span className="text-gray-500 italic">empty</span>}
      </div>
    </div>
  );
}

function VariantsTab({
  events,
  bundle,
}: {
  events: { index: number; label: string; text: string }[];
  bundle: VariantBundle | null;
}) {
  if (events.length === 0) {
    return <EmptyState message="Click Simulate Variants to start" />;
  }

  const scored: ScoredResponse[] = bundle
    ? [bundle.primary, ...bundle.alternatives]
    : [];

  return (
    <div className="flex flex-col gap-3">
      {events.map((ev, i) => {
        const score = scored[i];
        return (
          <div
            key={ev.index}
            className="bg-surface border border-border rounded p-3"
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-accent text-xs font-bold">
                #{ev.index + 1}
              </span>
              <span className="text-gray-400 text-xs">{ev.label}</span>
              {!bundle && (
                <span className="ml-auto">
                  <span className="inline-block w-2 h-2 rounded-full bg-accent animate-pulse" />
                </span>
              )}
            </div>
            <div className="text-sm whitespace-pre-wrap text-gray-300 mb-2 leading-relaxed">
              {ev.text}
            </div>
            {score && (
              <div className="flex gap-3 text-xs text-gray-500">
                <span>
                  Quality:{" "}
                  <span className="text-accent">
                    {score.quality.toFixed(2)}
                  </span>
                </span>
                <span>
                  Diversity:{" "}
                  <span className="text-blue-400">
                    {score.diversity.toFixed(3)}
                  </span>
                </span>
                <span>
                  Uncertainty:{" "}
                  <span className="text-yellow-400">
                    {score.uncertainty.toFixed(2)}
                  </span>
                </span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function PossibilityMapTab({
  result,
}: {
  result: PossibilityMapType | null;
}) {
  if (!result) {
    return <EmptyState message="Click Explore Space to generate a possibility map" />;
  }

  const coords: UmapCoord[] = result.top_paths.map((p, i) => ({
    text: p.text,
    x: i * 1.5,
    y: Math.sin(i) * 2,
    cluster_id: 0,
    config_label: `path ${i + 1} (${p.log_prob.toFixed(2)})`,
  }));

  return <PossibilityMap coords={coords} />;
}

function JsonTab({
  response,
  variantBundle,
  exploreResult,
}: {
  response: ResponseResult | null;
  variantBundle: VariantBundle | null;
  exploreResult: PossibilityMapType | null;
}) {
  const data = { response, variantBundle, exploreResult };
  return (
    <pre className="text-xs text-gray-400 whitespace-pre-wrap break-all">
      {JSON.stringify(data, null, 2) || "{}"}
    </pre>
  );
}

function Badge({ label, value }: { label: string; value: string }) {
  return (
    <span className="bg-surface border border-border rounded px-2 py-1">
      <span className="text-gray-500">{label}</span>{" "}
      <span className="text-accent">{value}</span>
    </span>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex items-center justify-center h-40 text-gray-600 text-sm">
      {message}
    </div>
  );
}
