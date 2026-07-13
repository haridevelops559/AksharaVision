import { NavLink } from "react-router-dom";
import "./Navbar.css";

export default function Navbar() {
  return (
    <nav className="navbar">

      <h2 className="logo">
        AksharaVision
      </h2>

      <div className="nav-links">

        <NavLink to="/">
          OCR
        </NavLink>

        <NavLink to="/monitoring">
          Monitoring
        </NavLink>

      </div>

    </nav>
  );
}