import { useState } from "react";
import { Routes, Route } from "react-router-dom";
import { X } from "lucide-react";
import { Sidebar } from "./components/Sidebar";
import { Header } from "./components/Header";
import { Dashboard } from "./pages/Dashboard";
import { Cameras } from "./pages/Cameras";
import { LiveMonitoring } from "./pages/LiveMonitoring";
import { VideoAnalysisPage } from "./pages/VideoAnalysisPage";
import { Violations } from "./pages/Violations";
import { EvidencePage } from "./pages/EvidencePage";
import { Analytics } from "./pages/Analytics";
import { Settings } from "./pages/Settings";
import { EventDetail } from "./pages/EventDetail";
import { AnalysisDetail } from "./pages/AnalysisDetail";

export default function App() {
  const [mobileNav, setMobileNav] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--bg-base)] text-[var(--text-primary)]">
      {/* desktop sidebar */}
      <div className="hidden lg:block">
        <Sidebar />
      </div>

      {/* mobile sidebar drawer */}
      {mobileNav && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={() => setMobileNav(false)} />
          <div className="absolute left-0 top-0 h-full">
            <div className="relative h-full">
              <Sidebar onNavigate={() => setMobileNav(false)} />
              <button
                onClick={() => setMobileNav(false)}
                className="absolute -right-10 top-3 flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--bg-panel)] text-[var(--text-secondary)]"
                aria-label="Close navigation"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <Header onMenuClick={() => setMobileNav(true)} />
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/cameras" element={<Cameras />} />
            <Route path="/live" element={<LiveMonitoring />} />
            <Route path="/analysis" element={<VideoAnalysisPage />} />
            <Route path="/analysis/:id" element={<AnalysisDetail />} />
            <Route path="/violations" element={<Violations />} />
            <Route path="/violations/:id" element={<EventDetail />} />
            <Route path="/evidence" element={<EvidencePage />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
