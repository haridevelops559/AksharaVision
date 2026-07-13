import { Routes, Route } from "react-router-dom";

import Navbar from "./components/Navbar";

import OCRPage from "./pages/OCRPage";
import MonitoringPage from "./pages/MonitoringPage";

function App() {
  return (
    <>
      <Navbar />

      <Routes>
        <Route path="/" element={<OCRPage />} />

        <Route
          path="/monitoring"
          element={<MonitoringPage />}
        />
      </Routes>
    </>
  );
}

export default App;