import { atom, selector } from "recoil";

/* a component for recoilstates and few componentized functions*/

//const to set profile image
export const DisplayImageAtom = atom({
  key: "displayImage",
  default: "",
});

// make cloudinary form for Register.js and Profile.js
export function makeFormData(e) {
  const img = e.target.files[0];

  const formData = new FormData();
  formData.append("file", img);
  formData.append("api_key", "276775976593738");
  formData.append("upload_preset", "h0iblbf5");
  formData.append("timestamp", (Date.now() / 1000) | 0);
  const config = {
    headers: { "Content-Type": "multipart/form-data" },
  };
  return { formData, config };
}

//shared between log.js and view.js
export const returneeState = atom({
  key: "returneeState",
  default: [],
});

//shared between log.js and view.js
export const questionsState = atom({
  key: "questionsState",
  default: [],
});

//shared between log.js and view.js
export const questionsSelector = selector({
  key: "questionsSelector",
  get: ({ get }) => get(questionsState),
  set: ({ set }, newValue) => {
    set(questionsState, newValue);
  },
});

//shared between log.js and view.js
const cum_date = new Date();
export const cumDateState = atom({
  key: "cumDateState",
  default: {
    cum_year: cum_date.getFullYear(),
    cum_month: cum_date.getMonth() + 1,
    cum_day: cum_date.getDate(),
  },
});

// Check if logged in before load page
// Send to main login page when not logged in
export function isAuthenticated(props, navigate) {
  if (props.profile.user_id == "") {
    alert(
      "not a valid path - please log in first. \n(note: redirection(F5) is not allowed) "
    );
    navigate("/");
  }
}
