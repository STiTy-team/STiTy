import { Navigate, Route, Routes } from "react-router"

import { AppSidebar } from "@/components/app-sidebar"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"
import { TooltipProvider } from "@/components/ui/tooltip"
import { ComparePage } from "@/pages/compare-page"
import { ConfigsPage } from "@/pages/configs-page"
import { DatasetsPage } from "@/pages/datasets-page"
import { QueuePage } from "@/pages/queue-page"
import { ReplayPage } from "@/pages/replay-page"
import { RunsPage } from "@/pages/runs-page"

const sidebarWasOpen = () => !document.cookie.split("; ").includes("sidebar_state=false")

export default function App() {
  return (
    <TooltipProvider>
      <SidebarProvider defaultOpen={sidebarWasOpen()}>
        <AppSidebar />
        <SidebarInset className="min-w-0">
          <main className="flex flex-1 flex-col">
            <Routes>
              <Route path="/" element={<Navigate to="/queue" replace />} />
              <Route path="/runs" element={<RunsPage />} />
              <Route path="/queue/:host?" element={<QueuePage />} />
              <Route path="/configs/:kind?" element={<ConfigsPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/replay" element={<ReplayPage />} />
              <Route path="/datasets/:tab?" element={<DatasetsPage />} />
            </Routes>
          </main>
        </SidebarInset>
      </SidebarProvider>
    </TooltipProvider>
  )
}
