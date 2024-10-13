
export const css = `
#potree_toolbar{
	position: absolute; 
	z-index: 10000; 
	left: 100px; 
	top: 0px;
	background: black;
	color: white;
	padding: 0.3em 0.8em;
	font-family: "system-ui";
	border-radius: 0em 0em 0.3em 0.3em;
	display: flex;
	transform: scale(1.3, 1.3);
	transform-origin: top left;
}

.potree_toolbar_label{
	text-align: center;
	font-size: smaller;
	opacity: 0.9;
	padding: 0.1em 0.5em 0.5em 0.5em;
}

.potree_toolbar_separator{
	background: white;
	padding: 0px;
	margin: 5px 10px;
	width: 1px;
}

.potree_toolbar_gradient_button{
	width: 2em;
	height: 3em;
}

 .potree_toolbar_gradient_button:hover {
	filter: brightness(120%);
}

.potree_toolbar_button{
	width: 3em;
	height: 3em;
	background-size: contain;
	background-repeat: no-repeat;
	background-color: transparent;
	border: none;
}

.potree_toolbar_button:hover{
	filter: brightness(120%);
	filter: drop-shadow(0px 0px 6px white) drop-shadow(0px 0px 6px white);
}

.potree_toolbar_dropdown_button{
	background: none;
	border: none;
	width: 100%;
	color: white;
	opacity: 0.5;
	font-weight: bold;
	background-position: center;
	background-repeat: no-repeat;
	background-size: 1.6em;
	height: 0.8em;
	margin-top: 0.2em;

}

.potree_toolbar_dropdown_button:hover{
	filter: brightness(120%);
	filter: drop-shadow(0px 0px 3px white);
	border: 1px solid rgba(255, 255, 255, 0.4);
	opacity: 1;
}

`;