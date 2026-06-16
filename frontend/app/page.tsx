import ParameterPanel from "@/components/ParameterPanel";
import ResponseViewer from "@/components/ResponseViewer";
import PossibilityMap from "@/components/PossibilityMap";

export default function Home() {
  return (
    <main className="min-h-screen p-8">
      <h1 className="text-3xl font-bold mb-8">RGVE</h1>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <ParameterPanel />
        <ResponseViewer />
        <PossibilityMap />
      </div>
    </main>
  );
}
