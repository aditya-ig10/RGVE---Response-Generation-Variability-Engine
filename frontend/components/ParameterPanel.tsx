"use client";

import { type Persona, type Domain } from "./types";

const PERSONAS: { value: Persona; label: string }[] = [
  { value: "precise", label: "Precise" },
  { value: "balanced", label: "Balanced" },
  { value: "creative", label: "Creative" },
  { value: "exploratory", label: "Exploratory" },
  { value: "expert", label: "Expert" },
];

const DOMAINS: { value: Domain; label: string }[] = [
  { value: "factual", label: "Factual" },
  { value: "general", label: "General" },
  { value: "technical", label: "Technical" },
  { value: "safe", label: "Safe" },
];

export interface ThetaParams {
  temperature: number;
  top_p: number;
  top_k: number;
  persona: Persona;
  domain: Domain;
  logit_bias: Record<string, number>;
  repetition_penalty: number;
}

interface Props {
  theta: ThetaParams;
  budget: number;
  loading: string | null;
  onChange: (patch: Partial<ThetaParams>) => void;
  onBudgetChange: (v: number) => void;
  onAction: (action: string) => void;
}

export default function ParameterPanel({
  theta,
  budget,
  loading,
  onChange,
  onBudgetChange,
  onAction,
}: Props) {
  const btn = (action: string, label: string) => (
    <button
      onClick={() => onAction(action)}
      disabled={loading !== null}
      className={`w-full px-3 py-2 rounded text-sm font-medium transition-all duration-150 disabled:opacity-40 ${
        loading === action
          ? "bg-accent/20 text-accent border border-accent/40 animate-pulse"
          : "bg-surface border border-border text-gray-300 hover:border-accent/60 hover:text-accent"
      }`}
    >
      {loading === action ? (
        <span className="inline-flex items-center gap-2">
          <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          {label}...
        </span>
      ) : (
        label
      )}
    </button>
  );

  return (
    <aside className="flex flex-col gap-5">
      <h1 className="text-2xl font-bold tracking-tight text-accent">RGVE</h1>
      <p className="text-xs text-gray-500 -mt-4">
        Response Generation Variation Explorer
      </p>

      <div className="flex flex-col gap-4">
        <SliderField
          label="Temperature"
          value={theta.temperature}
          min={0}
          max={2}
          step={0.05}
          display={theta.temperature.toFixed(2)}
          onChange={(v) => onChange({ temperature: v })}
        />
        <SliderField
          label="Top-p"
          value={theta.top_p}
          min={0}
          max={1}
          step={0.05}
          display={theta.top_p.toFixed(2)}
          onChange={(v) => onChange({ top_p: v })}
        />
        <SliderField
          label="Rep. Penalty"
          value={theta.repetition_penalty}
          min={1}
          max={2}
          step={0.05}
          display={theta.repetition_penalty.toFixed(2)}
          onChange={(v) => onChange({ repetition_penalty: v })}
        />
      </div>

      <SelectField
        label="Persona"
        value={theta.persona}
        options={PERSONAS}
        onChange={(v) => onChange({ persona: v as Persona })}
      />
      <SelectField
        label="Domain"
        value={theta.domain}
        options={DOMAINS}
        onChange={(v) => onChange({ domain: v as Domain })}
      />

      <div>
        <label className="block text-xs text-gray-500 mb-1">Explore Budget</label>
        <input
          type="number"
          min={50}
          max={500}
          step={50}
          value={budget}
          onChange={(e) => onBudgetChange(Number(e.target.value))}
          className="w-full text-sm"
        />
      </div>

      <div className="flex flex-col gap-2 pt-2 border-t border-border">
        {btn("generate", "Generate Single")}
        {btn("variants", "Simulate Variants")}
        {btn("explore", "Explore Space")}
      </div>
    </aside>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-500">{label}</span>
        <span className="text-accent">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="block text-xs text-gray-500 mb-1">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-surface border border-border text-gray-300 rounded px-2 py-1.5 text-sm font-mono outline-none focus:border-accent/60"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}
