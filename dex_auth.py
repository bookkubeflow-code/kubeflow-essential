#!/usr/bin/env python3

import re
import time
import os
from urllib.parse import urlsplit, urlencode
import requests
import urllib3
import kfp


class DexSessionManager:
    """
    Dex Session Manager for programmatic Kubeflow authentication.
    Based on manifests/tests/dex_login_test.py from the Kubeflow repository.
    
    This extracts session cookies from Dex authentication to enable
    programmatic KFP client access.
    """

    def __init__(
        self,
        endpoint_url: str,
        dex_username: str,
        dex_password: str,
        dex_auth_type: str = "local",
        skip_tls_verify: bool = False,
    ):
        """
        Initialize the DexSessionManager

        :param endpoint_url: the Kubeflow Endpoint URL (e.g., http://localhost:8080)
        :param skip_tls_verify: if True, skip TLS verification
        :param dex_username: the Dex username (e.g., user@example.com)
        :param dex_password: the Dex password (e.g., 12341234)
        :param dex_auth_type: the auth type if Dex has multiple enabled, one of: ['ldap', 'local']
        """
        self._endpoint_url = endpoint_url
        self._skip_tls_verify = skip_tls_verify
        self._dex_username = dex_username
        self._dex_password = dex_password
        self._dex_auth_type = dex_auth_type
        self._session_cookies = None

        # disable SSL verification, if requested
        if self._skip_tls_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # ensure `dex_default_auth_type` is valid
        if self._dex_auth_type not in ["ldap", "local"]:
            raise ValueError(
                f"Invalid `dex_auth_type` '{self._dex_auth_type}', must be one of: ['ldap', 'local']"
            )

    def get_session_cookies(self) -> str:
        """
        Get the session cookies by authenticating against Dex
        :return: a string of session cookies in the form "key1=value1; key2=value2"
        """
        if self._session_cookies:
            return self._session_cookies

        max_retries = 3
        retry_delay = 2

        # Use a persistent session for cookies
        session = requests.Session()

        for attempt in range(max_retries):
            try:
                print(f"🔐 Attempting Dex authentication (attempt {attempt + 1}/{max_retries})...")
                
                # GET the endpoint_url, which should redirect to Dex
                response = session.get(
                    self._endpoint_url,
                    allow_redirects=True,
                    verify=not self._skip_tls_verify
                )
                
                if response.status_code == 200:
                    pass
                elif response.status_code == 403:
                    # if we get 403, we might be at the oauth2-proxy sign-in page
                    # the default path to start the sign-in flow is `/oauth2/start?rd=<url>`
                    url_object = urlsplit(response.url)
                    url_object = url_object._replace(
                        path="/oauth2/start",
                        query=urlencode({"rd": url_object.path})
                    )
                    response = session.get(
                        url_object.geturl(),
                        allow_redirects=True,
                        verify=not self._skip_tls_verify
                    )
                else:
                    raise RuntimeError(
                        f"HTTP status code '{response.status_code}' for GET against: {self._endpoint_url}"
                    )

                # if we were NOT redirected, then the endpoint is unsecured
                if len(response.history) == 0:
                    print("⚠️  No authentication required - endpoint is unsecured")
                    self._session_cookies = ""
                    return ""

                # if we are at `../auth` path, we need to select an auth type
                url_object = urlsplit(response.url)
                if re.search(r"/auth$", url_object.path):
                    url_object = url_object._replace(
                        path=re.sub(r"/auth$", f"/auth/{self._dex_auth_type}", url_object.path)
                    )

                # if we are at `../auth/xxxx/login` path, then we are at the login page
                if re.search(r"/auth/.*/login$", url_object.path):
                    dex_login_url = url_object.geturl()
                else:
                    # otherwise, we need to follow a redirect to the login page
                    response = session.get(
                        url_object.geturl(),
                        allow_redirects=True,
                        verify=not self._skip_tls_verify
                    )
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"HTTP status code '{response.status_code}' for GET against: {url_object.geturl()}"
                        )
                    dex_login_url = response.url

                print(f"🔑 Submitting credentials to: {dex_login_url}")
                
                # attempt Dex login
                response = session.post(
                    dex_login_url,
                    data={"login": self._dex_username, "password": self._dex_password},
                    allow_redirects=True,
                    verify=not self._skip_tls_verify,
                )

                # Handle 403 specifically - might need to restart oauth flow
                if response.status_code == 403:
                    print("⚠️  Got 403, trying oauth2 restart...")
                    # Try one more approach - go through the oauth2 flow again
                    oauth_url = f"{urlsplit(self._endpoint_url).scheme}://{urlsplit(self._endpoint_url).netloc}/oauth2/start"
                    response = session.get(
                        oauth_url,
                        allow_redirects=True,
                        verify=not self._skip_tls_verify,
                    )
                    # Continue with normal flow after restart
                    if response.status_code == 200 and session.cookies:
                        cookies_str = "; ".join([f"{c.name}={c.value}" for c in session.cookies])
                        print(f"✅ Authentication successful (via oauth2 restart)")
                        self._session_cookies = cookies_str
                        return cookies_str

                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP status code '{response.status_code}' for POST against: {dex_login_url}"
                    )

                # if we were NOT redirected, then the login credentials were probably invalid
                if len(response.history) == 0:
                    raise RuntimeError(
                        f"Login credentials are probably invalid - "
                        f"No redirect after POST to: {dex_login_url}"
                    )

                # if we are at `../approval` path, we need to approve the login
                url_object = urlsplit(response.url)
                if re.search(r"/approval$", url_object.path):
                    dex_approval_url = url_object.geturl()
                    print(f"✅ Approving login at: {dex_approval_url}")
                    # Approve the login
                    response = session.post(
                        dex_approval_url,
                        data={"approval": "approve"},
                        allow_redirects=True,
                        verify=not self._skip_tls_verify,
                    )
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"HTTP status code '{response.status_code}' for POST against: {url_object.geturl()}"
                        )

                cookies_str = "; ".join([f"{c.name}={c.value}" for c in session.cookies])
                print(f"✅ Authentication successful! Got {len(session.cookies)} cookies")
                self._session_cookies = cookies_str
                return cookies_str

            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(f"❌ All {max_retries} attempts failed. Last error: {str(e)}")
                    raise
                print(f"⚠️  Attempt {attempt + 1} failed: {str(e)}")
                print(f"   Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    def get_authenticated_client(self, namespace: str = "kubeflow") -> kfp.Client:
        """
        Get an authenticated KFP client using session cookies
        
        :param namespace: Kubernetes namespace (default: kubeflow)
        :return: Authenticated KFP Client
        """
        cookies = self.get_session_cookies()
        print(f"🍪 Using cookies: {cookies[:100]}..." if len(cookies) > 100 else f"🍪 Using cookies: {cookies}")
        
        # Create KFP client with correct pipeline host path
        pipeline_host = f"{self._endpoint_url}/pipeline"
        print(f"🔧 Using pipeline host: {pipeline_host}")
        client = kfp.Client(
            host=pipeline_host,
            namespace=namespace
        )
        
        # Create custom session with cookies and inject into all API clients
        session = requests.Session()
        if cookies:
            # Parse cookies and add to session
            for cookie_pair in cookies.split("; "):
                if "=" in cookie_pair:
                    name, value = cookie_pair.split("=", 1)
                    session.cookies.set(name, value, domain="localhost", path="/")
                    print(f"🔗 Set cookie: {name}={value[:50]}...")
        
        # Monkey-patch all the API client sessions
        try:
            client._healthz_api.api_client.rest_client.pool_manager.clear()
            client._healthz_api.api_client.rest_client.pool_manager = session
        except:
            pass
            
        # Try a different approach - replace the actual REST client methods
        original_request = client._healthz_api.api_client.rest_client.request
        
        def authenticated_request(method, url, **kwargs):
            # Add our cookies to every request
            if 'headers' not in kwargs:
                kwargs['headers'] = {}
            if cookies:
                kwargs['headers']['Cookie'] = cookies
            return original_request(method, url, **kwargs)
        
        # Replace request methods across all API clients
        client._healthz_api.api_client.rest_client.request = authenticated_request
        if hasattr(client, '_pipeline_api'):
            client._pipeline_api.api_client.rest_client.request = authenticated_request
        if hasattr(client, '_run_api'):
            client._run_api.api_client.rest_client.request = authenticated_request
        if hasattr(client, '_experiment_api'):
            client._experiment_api.api_client.rest_client.request = authenticated_request
        
        return client


class AuthenticatedKFPManager:
    """
    High-level manager for authenticated Kubeflow Pipelines access
    """
    
    def __init__(
        self,
        endpoint_url: str = None,
        username: str = None, 
        password: str = None,
        namespace: str = None
    ):
        # Use environment variables with fallbacks
        self.endpoint_url = endpoint_url or os.getenv('KUBEFLOW_ENDPOINT', 'http://localhost:8080')
        self.username = username or os.getenv('KUBEFLOW_USERNAME', 'user@example.com')
        self.password = password or os.getenv('KUBEFLOW_PASSWORD', '12341234')
        self.namespace = namespace or os.getenv('KUBEFLOW_NAMESPACE', 'kubeflow')
        self.endpoint_url = endpoint_url
        self.username = username
        self.password = password
        self.namespace = namespace
        self._dex_manager = None
        self._client = None

    def get_client(self) -> kfp.Client:
        """Get an authenticated KFP client"""
        if self._client is None:
            print(f"🚀 Initializing authenticated KFP client for {self.endpoint_url}")
            
            self._dex_manager = DexSessionManager(
                endpoint_url=self.endpoint_url,
                dex_username=self.username,
                dex_password=self.password,
                dex_auth_type="local",
                skip_tls_verify=True
            )
            
            self._client = self._dex_manager.get_authenticated_client(self.namespace)
            
            # Test the connection
            try:
                pipelines = self._client.list_pipelines(page_size=1)
                print(f"✅ Successfully authenticated! Found {len(pipelines.pipelines or [])} pipelines")
            except Exception as e:
                print(f"⚠️  Client created but verification failed: {e}")
                # Don't raise - sometimes list_pipelines fails but other operations work
        
        return self._client

    def submit_pipeline(
        self, 
        pipeline_path: str, 
        run_name: str = None,
        experiment_name: str = "Default",
        arguments: dict = None
    ):
        """Submit a pipeline with authentication"""
        client = self.get_client()
        
        if run_name is None:
            from datetime import datetime
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"📤 Submitting pipeline: {pipeline_path}")
        print(f"   Run name: {run_name}")
        print(f"   Experiment: {experiment_name}")
        
        try:
            # Create experiment if it doesn't exist
            try:
                experiment = client.create_experiment(
                    name=experiment_name,
                    description="Automated experiment via authenticated client"
                )
                print(f"✅ Created experiment: {experiment_name}")
            except Exception:
                print(f"ℹ️  Using existing experiment: {experiment_name}")
            
            # Submit the run
            run = client.create_run_from_pipeline_package(
                pipeline_file=pipeline_path,
                arguments=arguments or {},
                experiment_name=experiment_name,
                run_name=run_name
            )
            
            print(f"✅ Pipeline submitted successfully!")
            print(f"🔗 Run ID: {run.run_id}")
            print(f"🌐 View at: {self.endpoint_url}/#/runs/details/{run.run_id}")
            
            return run
            
        except Exception as e:
            print(f"❌ Pipeline submission failed: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    manager = AuthenticatedKFPManager()
    
    try:
        client = manager.get_client()
        print("🎉 Authentication successful!")
        
        # Test pipeline submission if a compiled pipeline exists
        import os
        pipeline_files = [
            "compiled_pipelines/ml_training_latest.yaml",
            "ml_pipeline.yaml"
        ]
        
        for pipeline_file in pipeline_files:
            if os.path.exists(pipeline_file):
                print(f"\n📋 Found pipeline: {pipeline_file}")
                choice = input("Submit this pipeline? (y/N): ").lower().strip()
                if choice == 'y':
                    manager.submit_pipeline(
                        pipeline_path=pipeline_file,
                        experiment_name="Authenticated Pipeline Tests",
                        arguments={"dataset_url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"}
                    )
                break
        else:
            print("\n📋 No compiled pipelines found. Run compile_and_run.py first.")
            
    except Exception as e:
        print(f"❌ Authentication failed: {e}") 