import Axios from "axios";
import { useEffect, useState } from "react";
// import Topnav from "./nav";
import { hashutil } from "./hashutil.mjs";
import { useRef } from "react";

import { useRecoilState, useRecoilValue } from "recoil";
import {
  DisplayImageAtom,
  isAuthenticated,
  makeFormData,
} from "../model/states";
import { useNavigate } from "react-router-dom";

//profile page
export default function Profile(props) {
  const [img, setImg] = useState(props.profile.img);
  const [name, setName] = useState(props.profile.name);
  const [email, setEmail] = useState(props.profile.email);
  const [address1, setAddress1] = useState(props.profile.address1);
  const [address2, setAddress2] = useState(props.profile.address2);
  const [temp2, setTemp2] = useState({});
  const [tempConfig, setTempConfig2] = useState({});
  const [displayImage, setDisplayImage] = useRecoilState(DisplayImageAtom);
  const imageKeyRef = useRef("imageKeyRef");
  const navigate = useNavigate();

  const [endpoint, setEndpoint] = useState('image/upload');

  //post image on cloudinary and set profile image
  const setImage = (e) => {
    const { formData, config } = makeFormData(e);
    setTemp2(formData);
    setTempConfig2(config);

    Axios.post(
      `https://api.cloudinary.com/v1_1/dv2sy5qzq/${endpoint}`,
      formData,
      config
    ).then((res) => {
      console.log('file link', res.data.url);
      if (endpoint == 'image/upload') {
        setEndpoint('video/upload');
      } else {
        setEndpoint('image/upload');
      }
    });
  };

  //remove image
  const removeImage = () => {
    setDisplayImage("");
    setImg("");
  };

  const setUsername = (e) => {
    setName(e.target.value);
  };
  const setEmailAddress = (e) => {
    setEmail(e.target.value);
  };
  const setAddress1f = (e) => {
    setAddress1(e.target.value);
  };
  const setAddress2f = (e) => {
    setAddress2(e.target.value);
  };

  //log out and locate to main page
  const logout = () => {
    props.changeProfile({});
    document.location.href = "http://localhost:3000/";
  };
  const saveProfile = (e) => {
    Axios.put("https://diary30wooserver.web.app/api/users/parkinson", {
      // address_f: address1,
      user_id: props.profile.user_id,
      img: img
      
    }).then((res) => {
      alert("Your profile has been changed");
    });
  };
    
    //it should fetch and change
    //but for testing i will use just set function
    // if (/\S+@\S+\.\S+/.test(email)) {
    //   Axios.put("https://diary30wooserver.web.app/api/users", {
    //     user_id: props.profile.user_id,
    //     password: hashutil(
    //       props.profile.user_id,
    //       email,
    //       props.profile.password
    //     ),
    //     user_name: name,
    //     user_email: email,
    //     address_f: address1,
    //     address_l: address2,
    //     img: img,
    //     //address2: address2
    //   }).then((response) => {
    //     if (response.status != 200) {
    //       alert("Something went wrong in communicating DB!");
    //     } else {
    //       props.changeProfile({
    //         ...props.profile,
    //         name: name,
    //         email: email,
    //         address1: address1,
    //         address2: address2,
    //       });
    //       alert("Your profile has been changed");
    //     }
    //   });
    // } else {
    //   alert("Your email is not valid!\nYour profile has not been changed");
    // }

  //useEffect0 - check authentication before rendering
  useEffect(() => {
    isAuthenticated(props, navigate);
  }, []);

  return (
    <div id="profileWrapper">
      {/* <Topnav /> */}
      <form id="profileFormWrapper">
        <p>Dr.Parkinson</p>
        <div id="profileContent">
          <p id="profileContentTitle">Facial Test</p>
          <div id="profileUserInfo">
            {/* <img key={imageKeyRef} src={displayImage} alt="profile" /> */}
            <button
              id="profileImageSelector"
              type="button"
              class="btn btn-primary"
              data-toggle="modal"
              data-target="#exampleModal"
            >
              Upload Image
            </button>
            <div
              class="modal fade"
              id="exampleModal"
              tabindex="-1"
              role="dialog"
              aria-labelledby="exampleModalLabel"
              aria-hidden="true"
            >
              <div class="modal-dialog" role="document">
                <div class="modal-content">
                  <div class="modal-header">
                    <h5 class="modal-title" id="exampleModalLabel">
                    </h5>
                    <button
                      type="button"
                      class="close"
                      data-dismiss="modal"
                      aria-label="Close"
                    >
                      <span aria-hidden="true">&times;</span>
                    </button>
                  </div>
                  <div class="modal-body">
                    <p>Upload your file</p>
                    <input type="file" onChange={setImage} />
                  </div>
                  <div class="modal-footer">
                    <button
                      type="button"
                      class="btn btn-secondary"
                      data-dismiss="modal"
                    >
                      Close
                    </button>
                    <button type="button" class="btn btn-primary">
                      Save changes
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div id="profileContent">
          <p id="profileContentTitle">Walking Test</p>
          <div id="profileUserInfo">
            {/* <img key={imageKeyRef} src={displayImage} alt="profile" /> */}
            <button
              id="profileImageSelector"
              type="button"
              class="btn btn-primary"
              data-toggle="modal"
              data-target="#exampleModal"
            >
              Upload Video
            </button>
            <div
              class="modal fade"
              id="exampleModal"
              tabindex="-1"
              role="dialog"
              aria-labelledby="exampleModalLabel"
              aria-hidden="true"
            >
              <div class="modal-dialog" role="document">
                <div class="modal-content">
                  <div class="modal-header">
                    <h5 class="modal-title" id="exampleModalLabel">
                      Walking Test
                    </h5>
                    <button
                      type="button"
                      class="close"
                      data-dismiss="modal"
                      aria-label="Close"
                    >
                      <span aria-hidden="true">&times;</span>
                    </button>
                  </div>
                  <div class="modal-body">
                    <p>Set your walking video : </p>
                    <input type="file" onChange={setImage} />
                  </div>
                  <div class="modal-footer">
                    <button
                      type="button"
                      class="btn btn-secondary"
                      data-dismiss="modal"
                    >
                      Close
                    </button>
                    <button type="button" class="btn btn-primary">
                      Save changes
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        
         <div id="profileButtonWrapper">
          <button id="profileSubmit" type="button" onClick={saveProfile}>
            Detect
          </button>
          <button id="profileLogout" type="button" onClick={logout}>
            Logout
          </button>
        </div>
      </form>
    </div>
  );
}
