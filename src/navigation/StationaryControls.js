
import {Vector3, Matrix4, Box3, SphereMap} from "potree";
import {Potree, EventDispatcher} from "potree";
import * as TWEEN from "tween";

export class StationaryControls{

	constructor(element){

		this.element = element;
		this.yaw = 0;
		this.pitch = 0;
		this.pivot = new Vector3();
		this.world = new Matrix4();
		this.dispatcher = new EventDispatcher();

		this.sphereMap = new SphereMap();

		this.elFocusLabel = document.createElement("div");
		this.element.parentElement.append(this.elFocusLabel);
		this.elFocusLabel.innerText = `Currently viewing: nothing`;
		this.elFocusLabel.style.zIndex = 1000;
		this.elFocusLabel.style.color = "white";
		this.elFocusLabel.style.left = "10px";
		this.elFocusLabel.style.bottom = "10px";
		this.elFocusLabel.style.position = "absolute";
		this.elFocusLabel.style.fontSize = "2em";
		this.elFocusLabel.style.display = "none";
		this.elFocusLabel.style.textShadow = 
			`0.05em  0.05em 0.05em black, 
			 0.05em -0.05em 0.05em black, 
			-0.05em  0.05em 0.05em black, 
			-0.05em -0.05em 0.05em black`;

		this.dispatcher.add('mousemove', e => {

			let dragLeft = e.event.buttons === 1;

			if(dragLeft){
				let diffX = e.event.movementX;
				let diffY = e.event.movementY;

				let ux = diffX / this.element.width;
				let uy = diffY / this.element.height;

				this.yaw += 6 * ux;
				this.pitch += 6 * uy;
			}

			let {x, y} = e.mouse;

			Potree.pick(x, y, (result) => {

			});
		});

		this.dispatcher.add('mousewheel', e => {

			// TODO zooming

			// let diff = -Math.sign(e.delta);

			// if(diff > 0){
			// 	this.radius *= 1.1;
			// }else if(diff < 0){
			// 	this.radius /= 1.1;
			// }

		});

		this.dispatcher.add("focused", e => {
			console.log("focused!");

			this.elFocusLabel.style.display = "block";

			Potree.instance.scene.root.children.push(this.sphereMap);
		});

		this.dispatcher.add("unfocusd", e => {

			console.log("unfocused!");
			this.elFocusLabel.style.display = "none";

			let root = Potree.instance.scene.root;
			root.children = root.children.filter(node => node !== this.sphereMap);
			
		});

	}

	setLabel(label){
		this.elFocusLabel.innerText = label;
	}

	set({yaw, pitch}){
		this.yaw = yaw ?? this.yaw;
		this.pitch = pitch ?? this.pitch;
	}

	getPosition(){
		return new Vector3().applyMatrix4(this.world);
	}

	update(delta){

		let flip = new Matrix4().set(
			1, 0, 0, 0,
			0, 0, 1, 0,
			0, -1, 0, 0,
			0, 0, 0, 1,
		);

		this.world.makeIdentity();
		// this.world.translate(0, 0, this.radius);
		this.world.multiplyMatrices(flip, this.world);
		this.world.rotate(Math.PI / 2 - this.pitch, new Vector3(1, 0, 0));
		this.world.rotate(-this.yaw, new Vector3(0, 1, 0));
		this.world.translate(this.pivot.x, this.pivot.z, -this.pivot.y);

		{
			let flip = new Matrix4().set(
				1, 0, 0, 0,
				0, 0, -1, 0,
				0, 1, 0, 0,
				0, 0, 0, 1,
			);

			this.world.multiplyMatrices(flip, this.world);
		}

	}

	toExpression(){

		let pivot = this.pivot;

		let str = `;
		controls.set(
			yaw: ${this.yaw},
			pitch: ${this.pitch},
			pivot: new Vector3(${pivot.x}, ${pivot.y}, ${pivot.z}),
		);
		`;

		return str;
	}



};