import React, { Link } from "react-router-dom";
import { useState, useEffect } from "react";
import Axios from "axios";
import { hashutil } from "./hashutil.mjs";

import { useRecoilState, useRecoilValue } from "recoil";
import { DisplayImageAtom, makeFormData } from "../model/states";

//register page
export default function Register(props) {
  const [displayImage, setDisplayImage] = useRecoilState(DisplayImageAtom);
  const [value, setValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [profdata, setProfdata] = useState({
    user_id: "",
    password: "",
    user_name: "",
    user_email: "",
    address_f: "",
    address_l: "",
    img: "",
  });
  // contain formData and config, later sent to cloudinary
  const [hasFile, setHasFile] = useState(0);
  const [temp2, setTemp2] = useState({});
  const [tempConfig, setTempConfig] = useState({});
  const {
    user_id,

    password,

    user_name,

    user_email,

    address_f,

    address_l,

    img,
  } = profdata;

  const onDrop = async (e) => {
    const { formData, config } = makeFormData(e);
    //
    setHasFile(1);
    setTemp2(formData);
    setTempConfig(config);
  };

  //push the input values to db * exept "pw"
  const setTextid = (e) => {
    const { name, value } = e.target;

    setProfdata({
      ...profdata,
      [name]: value,
    });
  };

  //push the input val to db * only for "pw"
  const setTextid2 = (e) => {
    const { name, value } = e.target;
    setValue(e.target.value);

    setProfdata({
      ...profdata,
      [name]: value,
    });
  };

  //disable register button while register button is in process
  async function waitForRegister() {
    try {
      setIsLoading(true);
      await isRegister();
    } finally {
      setIsLoading(false);
    }
  }

  //check rules for id,pw... and push to server
  async function isRegister() {
    switch ("") {
      case user_id:
        alert("Put id!");
        break;
      case password:
        alert("Put Password!");
        break;
      case user_name:
        alert("Put name!");
        break;
      case user_email:
        alert("Put email!");
        break;
      case address_f:
        alert("Put first address!");
        break;
      case address_l:
        alert("Put last address!");
        break;
      default:
        let temp1 = 0;
        if (!/\S+@\S+\.\S+/.test(profdata.user_email)) {
          alert("Your email is not in valid form!");
          temp1 = 1;
        } else if (
          !/^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[a-zA-Z]).{8,}$/.test(
            profdata.password
          )
        ) {
          alert("Your password is not in valid form!");
          temp1 = 1;
        } else {
          let temp = 0;
          if (temp1 === 0) {
            await Axios.get("https://diary30wooserver.web.app/api/users").then(
              (response) => {
                for (var i in response.data) {
                  if (response.data[i] == user_id) {
                    break;
                  } else if (
                    i == response.data.length - 1 &&
                    response.data[i] != user_id
                  ) {
                    temp = 1;
                  }
                }
                if (temp === 1) {
                  if (hasFile == 1) {
                    Axios.post(
                      "https://api.cloudinary.com/v1_1/dl1bnuva1/image/upload",
                      temp2,
                      tempConfig
                    ).then((res) => {
                      Axios.post("https://diary30wooserver.web.app/api/users", {
                        user_id: profdata.user_id,
                        password: hashutil(user_id, user_email, password),
                        user_name: profdata.user_name,
                        user_email: profdata.user_email,
                        address_f: profdata.address_f,
                        address_l: profdata.address_l,
                        img: res.data.url,
                      }).then(() => {
                        setDisplayImage(res.data.url);
                        alert("Success Register!");
                        document.location.href = "http://localhost:3000/";
                      });
                    });
                  }
                  if (hasFile == 0) {
                    Axios.post("https://diary30wooserver.web.app/api/users", {
                      user_id: profdata.user_id,
                      password: hashutil(user_id, user_email, password),
                      user_name: profdata.user_name,
                      user_email: profdata.user_email,
                      address_f: profdata.address_f,
                      address_l: profdata.address_l,
                      img: profdata.img,
                    }).then(() => {
                      setDisplayImage(profdata.img);
                      alert("Success Register!");
                      document.location.href = "http://localhost:3000/";
                    });
                  }
                } else if (temp === 0) {
                  alert("Same ID already exist!");
                }
              }
            );
          }
        }
    }
  }

  // to login with "Enter key"
  const handleKeyDown = (event) => {
    if (event.keyCode === 13) {
      event.preventDefault();
      document.getElementById("registerButton").click();
    }
  };

  // to login with "Enter key"
  useEffect(() => {
    const registerForm = document.getElementById("registerWrapper");
    if (!registerForm) return;

    registerForm.addEventListener("keyup", handleKeyDown);
    return () => {
      if (!registerForm) return;
      registerForm.removeEventListener("keyup", handleKeyDown);
    };
  }, []);

  return (
    <div id="registerWrapper">
      <inner>
        <p>Register</p>
        <div>
          <nav>
            <ul>
              <li>
                <div>
                  <p>Set your ID : </p>
                  <input
                    id="user_id"
                    name="user_id"
                    value={user_id}
                    type="text"
                    onChange={setTextid}
                  />
                </div>
              </li>
              <li>
                <div>
                  <p>Set your password : </p>
                  <label class={`input-label ${value ? "has-value" : ""}`}>
                    At least 8chars, uppercase + lowercase + number{" "}
                  </label>
                  <input
                    name="password"
                    defaultValue={password}
                    type="password"
                    onChange={setTextid2}
                  />
                </div>
              </li>
              <li>
                <div>
                  <p>Set your name : </p>
                  <input
                    name="user_name"
                    value={user_name}
                    type="text"
                    onChange={setTextid}
                  />
                </div>
              </li>
              <li>
                <div>
                  <p>Set your email : </p>
                  <input
                    name="user_email"
                    value={user_email}
                    type="text"
                    onChange={setTextid}
                  />
                </div>
              </li>
              <li id="registerAddress">
                <div>
                  <p>Set your address : </p>
                  <div>
                    <input
                      name="address_f"
                      value={address_f}
                      type="text"
                      onChange={setTextid}
                    />
                  </div>
                  <div>
                    <input
                      name="address_l"
                      value={address_l}
                      type="text"
                      onChange={setTextid}
                    />
                  </div>
                </div>
              </li>
              <li>
                <div>
                  <p>Set your profile image : </p>
                  <input type="file" onChange={onDrop} />
                </div>
              </li>
            </ul>
            <div id="registerButtonWrapper">
              <button
                id="registerButton"
                onClick={waitForRegister}
                disabled={isLoading}
              >
                Register
              </button>
            </div>
          </nav>
        </div>
      </inner>
    </div>
  );
}
