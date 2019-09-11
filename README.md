# Running 

In order to run you need to have `python 3.7` and `pip` installed.

You need to set your Auth0 Domain and the API's audience as environment variables with the following names
respectively: `AUTH0_DOMAIN` and `API_IDENTIFIER`, which is the audience of your API. 

For that, if you just create a file named `.env` in the directory and set the values like the following,
the app will just work:

```bash
# .env file
AUTH0_DOMAIN=example.auth0.com
API_IDENTIFIER=https://quickstart/api
```

Once you've set those 2 environment variables:

1. Install the needed dependencies with `pip install -r requirements.txt`
2. Start the server with `ilaplace$ hypercorn -b 0.0.0.0:8000 server:APP`

# Testing the API

You can then try to do a GET to [http://localhost:3010/api/private](http://localhost:3010/api/private) which will
throw an error if you don't send an access token signed with RS256 with the appropriate issuer and audience in the
Authorization header. 

# Running the example with Docker

In order to run the sample with [Docker](https://www.docker.com/) you need to add the `AUTH0_DOMAIN` and `API_ID`
to the `.env` filed as explained [previously](#running-the-example) and then

1. Execute in command line `sh exec.sh` to run the Docker in Linux, or `.\exec.ps1` to run the Docker in Windows.
2. Try calling [http://localhost:3010/api/public](http://localhost:3010/api/public)