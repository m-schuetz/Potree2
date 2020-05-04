

import {Vector3} from "./src/math/Vector3.js";
import {Quaternion} from "./src/math/Quaternion.js";
import {Matrix4} from "./src/math/Matrix4.js";


{
	let q = new Quaternion(0, 0, 0, 1).setFromEuler(1, 2, 3);
	let v = new Vector3(0, 1, 0).applyQuaternion(q);

	console.log(q);
	console.log(v);
}

{
	let yaw = Math.PI / 4;
	let pitch = 0.1;

	let qYaw = new Quaternion().setFromEuler(0, 0, yaw);
	let qPitch = new Quaternion().setFromEuler(pitch, 0, 0);
	let orientation = new Quaternion().multiplyQuaternions(qYaw, qPitch);
	let forward = new Vector3(0, 1, 0).applyQuaternion(orientation);
	let right = new Vector3(1, 0, 0).applyQuaternion(orientation);
	let up = new Vector3(0, 0, 1).applyQuaternion(orientation);

	console.log("forward: ", forward);
	console.log("right: ", right);
	console.log("up: ", up);
}



