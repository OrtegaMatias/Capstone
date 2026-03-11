import { Navigate, Route, Routes } from 'react-router-dom';

import HomePage from './pages/HomePage';
import WeekPage from './pages/WeekPage';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/weeks/:weekId" element={<WeekPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
