import Axios from "axios";
import { useState, useEffect } from "react";
import React, { Link } from "react-router-dom";
import { useNavigate } from "react-router";
import { hashutil } from "./hashutil.mjs";

import { useRecoilState } from "recoil";
import { DisplayImageAtom } from "../model/states";

// Login page
export default function Login(props) {
  const [displayImage, setDisplayImage] = useRecoilState(DisplayImageAtom);
  const [isLoading, setIsLoading] = useState(false);
  const [login_id, setLogin] = useState("");
  const [password, setPassword] = useState("");
  const [avail, setAvail] = useState(false);
  const navigate = useNavigate();

  const setId = (e) => {
    setLogin(e.target.value);
  };
  const setPwd = (e) => {
    setPassword(e.target.value);
  };

  // Disable login button when net request is on process
  async function waitForLogin() {
    try {
      setIsLoading(true);
      await setLoginId();
    } finally {
      setIsLoading(false);
    }
  }

  // Set user profile and pass on to pages
  async function setLoginId() {
    await Axios.get(
      "https://diary30wooserver.web.app/api/users/" + login_id + ""
    ).then((response) => {
      if (response.data.length === 0) {
        alert("Your id is not found on DB");
      } else {
        if (
          response.data[0].password ==
          hashutil(login_id, response.data[0].user_email, password)
        ) {
          alert("Login Success!");
          props.ChangeProfile({
            password: password,
            user_id: response.data[0].user_id,
            profile: response.data[0].img,
            name: response.data[0].user_name,
            email: response.data[0].user_email,
            address1: response.data[0].address_f,
            address2: response.data[0].address_l,
            user_ref: response.data[0]._id,
          });
          setDisplayImage(() => response.data[0].img);
          navigate("/profile");
        } else {
          alert("ID or PWD is wrong!");
        }
      }
    });
  }

  // Make it possible to login with "Enter key"
  const handleKeyDown = (event) => {
    if (event.keyCode === 13) {
      event.preventDefault();
      document.getElementById("loginButton").click();
    }
  };

  // Add the "Enter key" method
  useEffect(() => {
    const loginForm = document.getElementById("loginWrapper");
    if (!loginForm) return;

    loginForm.addEventListener("keyup", handleKeyDown);
    return () => {
      if (!loginForm) return;
      loginForm.removeEventListener("keyup", handleKeyDown);
    };
  }, []);

  return (
    <div id="loginWrapper">
      <inner>
        <p>Login</p>
        <div>
          <nav>
            <ul>
              <li>
                <div>
                  <input
                    id="inputId"
                    placeholder="DIARY ID"
                    type="text"
                    onChange={setId}
                    autoComplete="off"
                  />
                </div>
                <br />
              </li>
            </ul>
            <ul>
              <li>
                <div>
                  <input
                    id="inputPw"
                    placeholder="PASSWORD"
                    type="password"
                    onChange={setPwd}
                    autoComplete="off"
                  />
                </div>
              </li>
            </ul>
            <div>
              <button
                id="loginButton"
                onClick={waitForLogin}
                disabled={isLoading}
              >
                Login
                <span class="material-symbols-outlined">login</span>
              </button>
              <Link to="/register">
                <button id="registerButton">
                  Don't have an account? Registe here ↗️
                </button>
              </Link>
            </div>
          </nav>
        </div>
      </inner>
    </div>
  );
}
