
import {Utils, EventDispatcher, KeyCodes, MouseCodes} from "potree";
import {Vector3} from "potree";

export class InputHandler extends EventDispatcher {
	constructor (potree) {
		super();

		this.potree = potree;
		this.renderer = potree.renderer;
		this.domElement = this.renderer.canvas;
		this.enabled = true;
		
		this.inputListeners = [];
		this.blacklist = new Set();

		this.drag = null;
		this.mouse = new Vector3(0, 0, 0);

		this.selection = [];

		this.hoveredElement = null;
		this.pressedKeys = {};

		this.wheelDelta = 0;

		this.speed = 1;

		this.logMessages = false;

		if (this.domElement.tabIndex === -1) {
			this.domElement.tabIndex = 2222;
		}

		this.domElement.addEventListener('contextmenu', (event) => { event.preventDefault(); }, false);
		this.domElement.addEventListener('click', this.onMouseClick.bind(this), false);
		this.domElement.addEventListener('mousedown', this.onMouseDown.bind(this), false);
		this.domElement.addEventListener('mouseup', this.onMouseUp.bind(this), false);
		this.domElement.addEventListener('mousemove', this.onMouseMove.bind(this), false);
		this.domElement.addEventListener('mousewheel', this.onMouseWheel.bind(this), false);
		this.domElement.addEventListener('DOMMouseScroll', this.onMouseWheel.bind(this), false); // Firefox
		this.domElement.addEventListener('dblclick', this.onDoubleClick.bind(this));
		this.domElement.addEventListener('touchstart', this.onTouchStart.bind(this));
		this.domElement.addEventListener('touchend', this.onTouchEnd.bind(this));
		this.domElement.addEventListener('touchmove', this.onTouchMove.bind(this));

		
		// this.domElement.addEventListener('keydown', this.onKeyDown.bind(this));
		// this.domElement.addEventListener('keyup', this.onKeyUp.bind(this));
		window.addEventListener('keydown', this.onKeyDown.bind(this));
		window.addEventListener('keyup', this.onKeyUp.bind(this));
	}

	addInputListener (listener) {
		this.inputListeners.push(listener);
	}

	removeInputListener (listener) {
		this.inputListeners = this.inputListeners.filter(e => e !== listener);
	}

	getSortedListeners(){
		return this.inputListeners.sort( (a, b) => {
			let ia = (a.importance !== undefined) ? a.importance : 0;
			let ib = (b.importance !== undefined) ? b.importance : 0;

			return ib - ia;
		});
	}

	onTouchStart (e) {
		if (this.logMessages) console.log(this.constructor.name + ': onTouchStart');

		e.preventDefault();

		if (e.touches.length === 1) {
			let rect = this.domElement.getBoundingClientRect();
			let x = e.touches[0].pageX - rect.left;
			let y = e.touches[0].pageY - rect.top;
			this.mouse.set(x, y);

			this.startDragging(null);
		}

		
		for (let inputListener of this.getSortedListeners()) {
			inputListener.dispatch(e.type, {
				touches: e.touches,
				changedTouches: e.changedTouches
			});
		}
	}

	onTouchEnd (e) {
		if (this.logMessages) console.log(this.constructor.name + ': onTouchEnd');

		e.preventDefault();

		for (let inputListener of this.getSortedListeners()) {
			inputListener.dispatch("drop", {
				drag: this.drag,
				viewer: this.viewer
			});
		}

		this.drag = null;

		for (let inputListener of this.getSortedListeners()) {
			inputListener.dispatch(e.type, {
				touches: e.touches,
				changedTouches: e.changedTouches
			});
		}
	}

	onTouchMove (e) {
		if (this.logMessages) console.log(this.constructor.name + ': onTouchMove');

		e.preventDefault();

		if (e.touches.length === 1) {
			let rect = this.domElement.getBoundingClientRect();
			let x = e.touches[0].pageX - rect.left;
			let y = e.touches[0].pageY - rect.top;
			this.mouse.set(x, y);

			if (this.drag) {
				this.drag.mouse = 1;

				this.drag.lastDrag.x = x - this.drag.end.x;
				this.drag.lastDrag.y = y - this.drag.end.y;

				this.drag.end.set(x, y);

				if (this.logMessages) console.log(this.constructor.name + ': drag: ');
				for (let inputListener of this.getSortedListeners()) {
					inputListener.dispatch("drag", {
						drag: this.drag,
						viewer: this.viewer
					});
				}
			}
		}

		for (let inputListener of this.getSortedListeners()) {
			inputListener.dispatch(e.type, {
				touches: e.touches,
				changedTouches: e.changedTouches
			});
		}

		// DEBUG CODE
		// let debugTouches = [...e.touches, {
		//	pageX: this.domElement.clientWidth / 2,
		//	pageY: this.domElement.clientHeight / 2}];
		// for(let inputListener of this.getSortedListeners()){
		//	inputListener.dispatch({
		//		type: e.type,
		//		touches: debugTouches,
		//		changedTouches: e.changedTouches
		//	});
		// }
	}

	onKeyDown (e) {
		if (this.logMessages) console.log(this.constructor.name + ': onKeyDown');

		// DELETE
		// if (e.keyCode === KeyCodes.DELETE && this.selection.length > 0) {
		// 	this.dispatch("delete", {
		// 		selection: this.selection
		// 	});

		// 	this.deselectAll();
		// }

		this.dispatch("keydown", {
			keyCode: e.keyCode,
			event: e
		});

		// for(let l of this.getSortedListeners()){
		//	l.dispatch({
		//		type: "keydown",
		//		keyCode: e.keyCode,
		//		event: e
		//	});
		// }

		this.pressedKeys[e.code] = true;

		// e.preventDefault();
	}

	onKeyUp (e) {
		if (this.logMessages) console.log(this.constructor.name + ': onKeyUp');

		delete this.pressedKeys[e.code];

		e.preventDefault();
	}

	onDoubleClick (e) {
		if (this.logMessages) console.log(this.constructor.name + ': onDoubleClick');

		let consumed = false;

		if(this.hoveredElement){
			if (this.hoveredElement?.dispatcher?.listeners?.get("dblclick")) {
				this.hoveredElement.dispatcher.dispatch("dblclick", {
					mouse: this.mouse,
					object: this.hoveredElement.object
				});
				consumed = true;
			}
		}

		if (!consumed) {
			for (let inputListener of this.getSortedListeners()) {
				inputListener.dispatch("dblclick", {
					mouse: this.mouse,
					object: null
				});
			}
		}

		e.preventDefault();
	}

	onMouseClick (e) {
		if (this.logMessages) console.log(this.constructor.name + ': onMouseClick');

		e.preventDefault();
	}

	onMouseDown (e) {
		if (this.logMessages) console.log(this.constructor.name + ': onMouseDown');

		e.preventDefault();

		let consumed = false;
		let consume = () => { consumed = true; };

		if(this.hoveredElement){
			if (this.hoveredElement?.dispatcher?.listeners?.get("mousedown")) {
				this.hoveredElement.dispatcher.dispatch("mousedown", {
					mouse: this.mouse,
					object: this.hoveredElement.object,
					consume
				});
				consumed = true;
			}
		}

		if (!consumed) {
			for (let inputListener of this.getSortedListeners()) {
				inputListener.dispatch("mousedown", {
					mouse: this.mouse,
					object: null
				});
			}
		}

		if (!this.drag) {

			let target = this.hoveredElements[0];
			let hasDragEvent = target?.object?.dispatcher?.listeners?.get("drag")?.length > 0;

			if (target && hasDragEvent) {
				this.startDragging(target.object, {location: target.point});
			} else {
				this.startDragging(null);
			}
		}
	}

	onMouseUp (e) {
		if (this.logMessages) console.log(this.constructor.name + ': onMouseUp');

		e.preventDefault();

		let noMovement = this.getNormalizedDrag().length() === 0;
		
		let consumed = false;
		let consume = () => { return consumed = true; };
		if (this.hoveredElements.length === 0) {
			for (let inputListener of this.getSortedListeners()) {
				inputListener.dispatch("mouseup", {
					viewer: this.viewer,
					mouse: this.mouse,
					consume: consume,
					event: e,
				});

				if(consumed){
					break;
				}
			}
		}else{
			let hovered = this.hoveredElements
				.map(e => e.object)
				.find(e => (e._listeners && e._listeners['mouseup']));
			if(hovered){
				hovered.dispatch("mouseup", {
					mouse: this.mouse,
					consume: consume,
					hovered,
					event: e,
				});
			}else{
				for (let inputListener of this.getSortedListeners()) {
					inputListener.dispatch("mouseup", {
						viewer: this.viewer,
						mouse: this.mouse,
						consume: consume,
						event: e,
					});

					if(consumed){
						break;
					}
				}
			}
		}

		if (this.drag) {
			if (this.drag.object) {
				if (this.logMessages) console.log(`${this.constructor.name}: drop ${this.drag.object.name}`);
				this.drag.object.dispatcher.dispatch("drop", {
					drag: this.drag,
					viewer: this.viewer

				});
			} else {
				for (let inputListener of this.getSortedListeners()) {
					inputListener.dispatch("drop", {
						drag: this.drag,
						viewer: this.viewer
					});
				}
			}

			// check for a click
			let target = this.hoveredElements[0];
			let clicked = target?.object === this.drag.object;
			if(clicked){
				if (this.logMessages) console.log(`${this.constructor.name}: click ${this.drag.object.name}`);
				this.drag.object.dispatcher.dispatch("click", {
					viewer: this.viewer,
					consume: consume,
					hovered: target,
				});
			}

			this.drag = null;
		}

		// if(!consumed){
		// 	if (e.button === MouseCodes.LEFT) {
		// 		if (noMovement) {
		// 			let selectable = this.hoveredElements
		// 				.find(el => el.object._listeners && el.object._listeners['select']);

		// 			if (selectable) {
		// 				selectable = selectable.object;

		// 				if (this.isSelected(selectable)) {
		// 					this.selection
		// 						.filter(e => e !== selectable)
		// 						.forEach(e => this.toggleSelection(e));
		// 				} else {
		// 					this.deselectAll();
		// 					this.toggleSelection(selectable);
		// 				}
		// 			} else {
		// 				this.deselectAll();
		// 			}
		// 		}
		// 	} else if ((e.button === MouseCodes.RIGHT) && noMovement) {
		// 		this.deselectAll();
		// 	}
		// }
	}

	onMouseMove (e) {
		e.preventDefault();

		let rect = this.domElement.getBoundingClientRect();
		let x = e.clientX - rect.left;
		let y = e.clientY - rect.top;
		this.mouse.set(x, y);

		for (let inputListener of this.getSortedListeners()) {
			inputListener.dispatch("mousemove", {
				mouse: this.mouse,
				event: e,
				inputhandler: this,
			});
		}
	}
	
	onMouseWheel(e){
		if(!this.enabled) return;

		if(this.logMessages) console.log(this.constructor.name + ": onMouseWheel");
		
		e.preventDefault();

		let delta = 0;
		if (e.wheelDelta !== undefined) { // WebKit / Opera / Explorer 9
			delta = e.wheelDelta;
		} else if (e.detail !== undefined) { // Firefox
			delta = -e.detail;
		}

		let ndelta = Math.sign(delta);

		for (let inputListener of this.getSortedListeners()) {
			inputListener.dispatch("mousewheel", {
				delta: ndelta,
				object: null,
				event: e,
			});
		}
		
	}

	startDragging (object, args = null) {

		let name = object ? object.name : "no name";
		if (this.logMessages) console.log(`${this.constructor.name}: startDragging: '${name}'`);

		this.drag = {
			start: this.mouse.clone(),
			end: this.mouse.clone(),
			lastDrag: new Vector3(0, 0, 0),
			// startView: this.scene.view.clone(),
			object: object
		};

		if (args) {
			for (let key of Object.keys(args)) {
				this.drag[key] = args[key];
			}
		}
	}

	update (delta) {

	}

	getNormalizedDrag () {
		if (!this.drag) {
			return new Vector3(0, 0, 0);
		}

		let diff = new Vector3().sub(this.drag.end, this.drag.start);

		diff.x = diff.x / this.domElement.clientWidth;
		diff.y = diff.y / this.domElement.clientHeight;

		return diff;
	}

	getNormalizedLastDrag () {
		if (!this.drag) {
			return new Vector3(0, 0, 0);
		}

		let lastDrag = this.drag.lastDrag.clone();

		lastDrag.x = lastDrag.x / this.domElement.clientWidth;
		lastDrag.y = lastDrag.y / this.domElement.clientHeight;

		return lastDrag;
	}
}