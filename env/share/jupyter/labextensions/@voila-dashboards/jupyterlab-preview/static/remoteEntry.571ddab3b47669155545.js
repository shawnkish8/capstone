var _JUPYTERLAB;(()=>{"use strict";var e,r,t,a,n,o,i,u,l,s,d,f,p,c,h,v,b,y,g,m,w,j={214:(e,r,t)=>{var a={"./index":()=>t.e(925).then((()=>()=>t(925))),"./extension":()=>t.e(925).then((()=>()=>t(925))),"./style":()=>t.e(776).then((()=>()=>t(776)))},n=(e,r)=>(t.R=r,r=t.o(a,e)?a[e]():Promise.resolve().then((()=>{throw new Error('Module "'+e+'" does not exist in container.')})),t.R=void 0,r),o=(e,r)=>{if(t.S){var a=t.S.default,n="default";if(a&&a!==e)throw new Error("Container initialization failed as it has already been initialized with a different share scope");return t.S[n]=e,t.I(n,r)}};t.d(r,{get:()=>n,init:()=>o})}},S={};function k(e){var r=S[e];if(void 0!==r)return r.exports;var t=S[e]={id:e,exports:{}};return j[e](t,t.exports,k),t.exports}k.m=j,k.c=S,k.n=e=>{var r=e&&e.__esModule?()=>e.default:()=>e;return k.d(r,{a:r}),r},k.d=(e,r)=>{for(var t in r)k.o(r,t)&&!k.o(e,t)&&Object.defineProperty(e,t,{enumerable:!0,get:r[t]})},k.f={},k.e=e=>Promise.all(Object.keys(k.f).reduce(((r,t)=>(k.f[t](e,r),r)),[])),k.u=e=>e+"."+{776:"5a65bd8d418bae34dfbf",925:"d3dbd2634726a35062f9"}[e]+".js?v="+{776:"5a65bd8d418bae34dfbf",925:"d3dbd2634726a35062f9"}[e],k.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"==typeof window)return window}}(),k.o=(e,r)=>Object.prototype.hasOwnProperty.call(e,r),e={},r="@voila-dashboards/jupyterlab-preview:",k.l=(t,a,n,o)=>{if(e[t])e[t].push(a);else{var i,u;if(void 0!==n)for(var l=document.getElementsByTagName("script"),s=0;s<l.length;s++){var d=l[s];if(d.getAttribute("src")==t||d.getAttribute("data-webpack")==r+n){i=d;break}}i||(u=!0,(i=document.createElement("script")).charset="utf-8",i.timeout=120,k.nc&&i.setAttribute("nonce",k.nc),i.setAttribute("data-webpack",r+n),i.src=t),e[t]=[a];var f=(r,a)=>{i.onerror=i.onload=null,clearTimeout(p);var n=e[t];if(delete e[t],i.parentNode&&i.parentNode.removeChild(i),n&&n.forEach((e=>e(a))),r)return r(a)},p=setTimeout(f.bind(null,void 0,{type:"timeout",target:i}),12e4);i.onerror=f.bind(null,i.onerror),i.onload=f.bind(null,i.onload),u&&document.head.appendChild(i)}},k.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},(()=>{k.S={};var e={},r={};k.I=(t,a)=>{a||(a=[]);var n=r[t];if(n||(n=r[t]={}),!(a.indexOf(n)>=0)){if(a.push(n),e[t])return e[t];k.o(k.S,t)||(k.S[t]={});var o=k.S[t],i="@voila-dashboards/jupyterlab-preview",u=[];switch(t){case"default":((e,r,t,a)=>{var n=o[e]=o[e]||{},u=n[r];(!u||!u.loaded&&(1!=!u.eager?a:i>u.from))&&(n[r]={get:()=>k.e(925).then((()=>()=>k(925))),from:i,eager:!1})})("@voila-dashboards/jupyterlab-preview","2.0.7")}return e[t]=u.length?Promise.all(u).then((()=>e[t]=1)):1}}})(),(()=>{var e;k.g.importScripts&&(e=k.g.location+"");var r=k.g.document;if(!e&&r&&(r.currentScript&&(e=r.currentScript.src),!e)){var t=r.getElementsByTagName("script");t.length&&(e=t[t.length-1].src)}if(!e)throw new Error("Automatic publicPath is not supported in this browser");e=e.replace(/#.*$/,"").replace(/\?.*$/,"").replace(/\/[^\/]+$/,"/"),k.p=e})(),t=e=>{var r=e=>e.split(".").map((e=>+e==e?+e:e)),t=/^([^-+]+)?(?:-([^+]+))?(?:\+(.+))?$/.exec(e),a=t[1]?r(t[1]):[];return t[2]&&(a.length++,a.push.apply(a,r(t[2]))),t[3]&&(a.push([]),a.push.apply(a,r(t[3]))),a},a=(e,r)=>{e=t(e),r=t(r);for(var a=0;;){if(a>=e.length)return a<r.length&&"u"!=(typeof r[a])[0];var n=e[a],o=(typeof n)[0];if(a>=r.length)return"u"==o;var i=r[a],u=(typeof i)[0];if(o!=u)return"o"==o&&"n"==u||"s"==u||"u"==o;if("o"!=o&&"u"!=o&&n!=i)return n<i;a++}},n=e=>{var r=e[0],t="";if(1===e.length)return"*";if(r+.5){t+=0==r?">=":-1==r?"<":1==r?"^":2==r?"~":r>0?"=":"!=";for(var a=1,o=1;o<e.length;o++)a--,t+="u"==(typeof(u=e[o]))[0]?"-":(a>0?".":"")+(a=2,u);return t}var i=[];for(o=1;o<e.length;o++){var u=e[o];i.push(0===u?"not("+l()+")":1===u?"("+l()+" || "+l()+")":2===u?i.pop()+" "+i.pop():n(u))}return l();function l(){return i.pop().replace(/^\((.+)\)$/,"$1")}},o=(e,r)=>{if(0 in e){r=t(r);var a=e[0],n=a<0;n&&(a=-a-1);for(var i=0,u=1,l=!0;;u++,i++){var s,d,f=u<e.length?(typeof e[u])[0]:"";if(i>=r.length||"o"==(d=(typeof(s=r[i]))[0]))return!l||("u"==f?u>a&&!n:""==f!=n);if("u"==d){if(!l||"u"!=f)return!1}else if(l)if(f==d)if(u<=a){if(s!=e[u])return!1}else{if(n?s>e[u]:s<e[u])return!1;s!=e[u]&&(l=!1)}else if("s"!=f&&"n"!=f){if(n||u<=a)return!1;l=!1,u--}else{if(u<=a||d<f!=n)return!1;l=!1}else"s"!=f&&"n"!=f&&(l=!1,u--)}}var p=[],c=p.pop.bind(p);for(i=1;i<e.length;i++){var h=e[i];p.push(1==h?c()|c():2==h?c()&c():h?o(h,r):!c())}return!!c()},i=(e,r)=>{var t=k.S[e];if(!t||!k.o(t,r))throw new Error("Shared module "+r+" doesn't exist in shared scope "+e);return t},u=(e,r)=>{var t=e[r];return(r=Object.keys(t).reduce(((e,r)=>!e||a(e,r)?r:e),0))&&t[r]},l=(e,r)=>{var t=e[r];return Object.keys(t).reduce(((e,r)=>!e||!t[e].loaded&&a(e,r)?r:e),0)},s=(e,r,t)=>"Unsatisfied version "+r+" of shared singleton module "+e+" (required "+n(t)+")",d=(e,r,t,a)=>{var n=l(e,t);return o(a,n)||"undefined"!=typeof console&&console.warn&&console.warn(s(t,n,a)),h(e[t][n])},f=(e,r,t)=>{var n=e[r];return(r=Object.keys(n).reduce(((e,r)=>!o(t,r)||e&&!a(e,r)?e:r),0))&&n[r]},p=(e,r,t,a)=>{var o=e[t];return"No satisfying version ("+n(a)+") of shared module "+t+" found in shared scope "+r+".\nAvailable versions: "+Object.keys(o).map((e=>e+" from "+o[e].from)).join(", ")},c=(e,r,t,a)=>{"undefined"!=typeof console&&console.warn&&console.warn(p(e,r,t,a))},h=e=>(e.loaded=1,e.get()),b=(v=e=>function(r,t,a,n){var o=k.I(r);return o&&o.then?o.then(e.bind(e,r,k.S[r],t,a,n)):e(r,k.S[r],t,a,n)})(((e,r,t,a)=>(i(e,t),h(f(r,t,a)||c(r,e,t,a)||u(r,t))))),y=v(((e,r,t,a)=>(i(e,t),d(r,0,t,a)))),g={},m={49:()=>y("default","@jupyterlab/ui-components",[1,3,1,13]),89:()=>y("default","@jupyterlab/application",[1,3,1,13]),168:()=>y("default","@lumino/signaling",[1,1,4,3]),271:()=>y("default","react",[1,17,0,1]),466:()=>y("default","@jupyterlab/settingregistry",[1,3,1,13]),595:()=>y("default","@jupyterlab/coreutils",[1,5,1,13]),770:()=>y("default","@jupyterlab/apputils",[1,3,1,13]),792:()=>y("default","@jupyterlab/mainmenu",[1,3,1,13]),797:()=>y("default","@lumino/coreutils",[1,1,5,3]),849:()=>y("default","@jupyterlab/notebook",[1,3,1,13]),875:()=>b("default","@jupyterlab/docregistry",[1,3,1,13])},w={925:[49,89,168,271,466,595,770,792,797,849,875]},k.f.consumes=(e,r)=>{k.o(w,e)&&w[e].forEach((e=>{if(k.o(g,e))return r.push(g[e]);var t=r=>{g[e]=0,k.m[e]=t=>{delete k.c[e],t.exports=r()}},a=r=>{delete g[e],k.m[e]=t=>{throw delete k.c[e],r}};try{var n=m[e]();n.then?r.push(g[e]=n.then(t).catch(a)):t(n)}catch(e){a(e)}}))},(()=>{var e={329:0};k.f.j=(r,t)=>{var a=k.o(e,r)?e[r]:void 0;if(0!==a)if(a)t.push(a[2]);else{var n=new Promise(((t,n)=>a=e[r]=[t,n]));t.push(a[2]=n);var o=k.p+k.u(r),i=new Error;k.l(o,(t=>{if(k.o(e,r)&&(0!==(a=e[r])&&(e[r]=void 0),a)){var n=t&&("load"===t.type?"missing":t.type),o=t&&t.target&&t.target.src;i.message="Loading chunk "+r+" failed.\n("+n+": "+o+")",i.name="ChunkLoadError",i.type=n,i.request=o,a[1](i)}}),"chunk-"+r,r)}};var r=(r,t)=>{var a,n,[o,i,u]=t,l=0;if(o.some((r=>0!==e[r]))){for(a in i)k.o(i,a)&&(k.m[a]=i[a]);u&&u(k)}for(r&&r(t);l<o.length;l++)n=o[l],k.o(e,n)&&e[n]&&e[n][0](),e[o[l]]=0},t=self.webpackChunk_voila_dashboards_jupyterlab_preview=self.webpackChunk_voila_dashboards_jupyterlab_preview||[];t.forEach(r.bind(null,0)),t.push=r.bind(null,t.push.bind(t))})();var E=k(214);(_JUPYTERLAB=void 0===_JUPYTERLAB?{}:_JUPYTERLAB)["@voila-dashboards/jupyterlab-preview"]=E})();