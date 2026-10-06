import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import InferencePage from "./features/inference/InferencePage";
import ModelDetailPage from "./features/models/ModelDetailPage";
import ModelsPage from "./features/models/ModelsPage";
import TrainingPage from "./features/training/TrainingPage";

const NAV = [
  { to: "/training", label: "Training" },
  { to: "/models", label: "Models" },
  { to: "/inference", label: "Inference" },
];

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>YOLOX Finetuning</h1>
        <nav>
          {NAV.map(({ to, label }) => (
            <NavLink key={to} to={to}>
              {label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/training" replace />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/models/:id" element={<ModelDetailPage />} />
          <Route path="/inference" element={<InferencePage />} />
          <Route path="*" element={<p>Page not found.</p>} />
        </Routes>
      </main>
    </div>
  );
}
