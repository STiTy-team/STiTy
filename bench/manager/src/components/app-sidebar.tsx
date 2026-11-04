import { ChartColumnIcon, DatabaseIcon, FileCogIcon, HistoryIcon, ListOrderedIcon } from "lucide-react";
import { NavLink, useLocation } from "react-router";

import stityLogo from "@/assets/stity-logo.png";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";

const PAGES = [
  { path: "/queue", label: "Queue", icon: ListOrderedIcon },
  { path: "/configs", label: "Configs", icon: FileCogIcon },
  { path: "/compare", label: "Compare", icon: ChartColumnIcon },
  { path: "/runs", label: "Runs", icon: HistoryIcon, also: ["/replay"] },
  { path: "/datasets", label: "Datasets", icon: DatabaseIcon },
];

export function AppSidebar() {
  const { pathname } = useLocation();
  return (
    <Sidebar>
      <SidebarHeader className="px-4 py-5">
        <a
          href="/"
          className="flex items-center gap-2"
          aria-label="STiTy Manager"
        >
          <img src={stityLogo} alt="STiTy" className="h-8 w-auto" />
          <div className="flex flex-col">
            <span className="pb-0.5 text-sm font-semibold tracking-wide text-sidebar-foreground/70">
              Experiment
            </span>
            <span className="pb-0.5 text-sm font-semibold tracking-wide text-sidebar-foreground/70">
              Manager
            </span>
          </div>
        </a>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {PAGES.map((page) => (
                <SidebarMenuItem key={page.path}>
                  <SidebarMenuButton asChild isActive={[page.path, ...(page.also ?? [])].some((path) => pathname.startsWith(path))}>
                    <NavLink to={page.path}>
                      <page.icon />
                      <span>{page.label}</span>
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarRail />
    </Sidebar>
  );
}
