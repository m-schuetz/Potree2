
const BUTTON_PRIMARY = 0b001; // left
const BUTTON_SECONDARY = 0b010; // right
const BUTTON_AUXILARY = 0b100; // middle

import {Vector3} from "../math/Vector3.js";
import {Quaternion} from "../math/Quaternion.js";

export class OrbitControls{

	constructor(element, camera){
		this.element = element;
		this.camera = camera;

		this.target = new Vector3(0, 0, 0);
		this.radius = 10.0;
		this.yaw = 0;
		this.pitch = 0;

		this.mouse = [0, 0];
		this.drag = null;

		element.addEventListener('mousedown', this.onMouseDown.bind(this), false);
		element.addEventListener('mouseup', this.onMouseUp.bind(this), false);
		element.addEventListener('mousemove', this.onMouseMove.bind(this), false);
		element.addEventListener('mousewheel', this.onMouseWheel.bind(this), false);
		element.addEventListener('DOMMouseScroll', this.onMouseWheel.bind(this), false); // Firefox
		element.addEventListener('contextmenu', (e) => {e.preventDefault()}, false);
	}

	onMouseDown(e){
		// console.log(e);
		e.preventDefault();
		e.stopImmediatePropagation();

		this.drag = {
			start: [...this.mouse],
			end: [...this.mouse],
			startYaw: this.yaw,
			startPitch: this.pitch,
			startTarget: this.target.clone(),
		};

		return false;
	}

	onMouseUp(e){
		// console.log(e);
		e.preventDefault();
		e.stopImmediatePropagation();

		this.drag = null;

		return false;
	}

	onMouseMove(e){
		e.preventDefault();
		e.stopImmediatePropagation();

		let rect = this.element.getBoundingClientRect();
		let x = e.clientX - rect.left;
		let y = e.clientY - rect.top;
		this.mouse = [x, y];

		if(this.drag){
			this.drag.end = [...this.mouse];
			this.onMouseDrag(e, this.drag);
		}
	}

	onMouseWheel(e){
		let delta = 0;
		if (e.wheelDelta !== undefined) { // WebKit / Opera / Explorer 9
			delta = e.wheelDelta;
		} else if (e.detail !== undefined) { // Firefox
			delta = -e.detail;
		}

		let ndelta = Math.sign(delta);

		this.radius = (-ndelta * 0.1 + 1) * this.radius;
	}

	onMouseDrag(e, drag){

		let {element} = this;

		let [width, height] = [element.clientWidth, element.clientHeight];
		let diagonal = Math.sqrt(width * width + height * height);

		let diffX = drag.end[0] - drag.start[0];
		let diffY = drag.end[1] - drag.start[1];
		
		if((e.buttons & BUTTON_PRIMARY) !== 0) { // orientation
			let rotateSpeed = 2 * Math.PI;
			let yaw = drag.startYaw - rotateSpeed * diffX / diagonal;
			let pitch = drag.startPitch - rotateSpeed * diffY / diagonal;

			this.yaw = yaw;
			this.pitch = pitch;
		}

		if((e.buttons & BUTTON_SECONDARY) !== 0){ // position
			console.log("translation");
			let qYaw = new Quaternion().setFromEuler(0, 0, this.yaw);
			let qPitch = new Quaternion().setFromEuler(this.pitch, 0, 0);
			let orientation = new Quaternion().multiplyQuaternions(qYaw, qPitch);

			let forward = new Vector3(0, 1, 0).applyQuaternion(orientation);
			let right = new Vector3(1, 0, 0).applyQuaternion(orientation);
			let up = new Vector3(0, 0, 1).applyQuaternion(orientation);

			let translation = new Vector3(0, 0, 0);

			let amountX = -this.radius * diffX / diagonal;
			translation.add(right.multiplyScalar(amountX));

			let amountY = this.radius * diffY / diagonal;
			translation.add(up.multiplyScalar(amountY));

			this.target.copy(drag.startTarget).add(translation);
		}
	}

	getDirection(){
		let qYaw = new Quaternion().setFromEuler(0, 0, this.yaw);
		let qPitch = new Quaternion().setFromEuler(this.pitch, 0, 0);
		let orientation = new Quaternion().multiplyQuaternions(qYaw, qPitch);

		let forward = new Vector3(0, 1, 0).applyQuaternion(orientation);

		return forward;
	}

	getPosition(){
		let dir = this.getDirection();
		dir.multiplyScalar(this.radius);

		let position = this.target.clone().sub(dir);

		return position;
	}

	getOrientation(){
		let qYaw = new Quaternion().setFromEuler(0, 0, this.yaw);
		let qPitch = new Quaternion().setFromEuler(this.pitch, 0, 0);
		let orientation = new Quaternion().multiplyQuaternions(qYaw, qPitch);

		return orientation;
	}

	update(delta){
		let {camera} = this;

		camera.position.copy(this.getPosition());
		camera.distance = this.radius;
		camera.orientation = this.getOrientation();
	}

}